"""Bounded background chunks with joined or paired CUDA graph dependencies."""
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import math
import torch


_captured = ContextVar('stepquant_captured_writes', default=None)


def attention_window():
    """Capture a release gate before attention work, including full attention."""
    tasks = _captured.get()
    if tasks is not None and tasks.attention_windows:
        tasks.window_count += 1
        tasks.release_window()


@dataclass
class WritebackChunk:
    pool: object
    ready: torch.cuda.Event
    tensors: tuple
    final: bool


class CapturedWrites(list):
    def __init__(self, attention_windows=False, joined=False):
        super().__init__()
        self.attention_windows = attention_windows
        self.joined = joined
        self.window_count = 0
        self.pending = deque()
        self.quota = 0
        self.layers = set()

    def release_window(self, count=None):
        # These are capture-time GPU release gates, not host-side waits.
        count = self.quota if count is None else count
        if not self.pending or not count:
            return
        ready = torch.cuda.Event(external=not self.joined)
        ready.record()
        for _ in range(min(count, len(self.pending))):
            pool, tensors, final = self.pending.popleft()
            super().append(WritebackChunk(pool, ready, tensors, final))
            if self.joined:
                with torch.cuda.stream(pool.writeback_stream):
                    ready.wait()
                    pool._finish_update(*tensors)


    def append(self, task):
        pool, *tensors = task
        if id(pool) in self.layers:
            raise ValueError('a layer may write its state only once per forward graph')
        self.layers.add(id(pool))
        batch = tensors[-1].numel()
        chunk = pool.writeback_chunk or batch
        for start in range(0, batch, chunk):
            end = min(start + chunk, batch)
            self.pending.append((pool, tuple(t[start:end] for t in tensors), end == batch))
        chunks = math.ceil(batch / chunk)
        early_kda=pool.plan.architecture=='kda' and (pool.storage=='byte' or getattr(pool,'early_writeback',False))
        # Each recurrent layer has two release points. Releasing fewer than
        # half its chunks at each point accumulates a long forward-end tail.
        early_quota=max(min(chunks,3),math.ceil(chunks/2))
        self.quota = (early_quota if early_kda else chunks) if self.attention_windows else math.ceil(chunks / 2)
        if not self.attention_windows or pool.plan.architecture=='gdn' or early_kda:
            self.release_window()



@contextmanager
def defer_writeback(pools, *, attention_windows=False, joined=False):
    pools = tuple(pools)
    tasks = CapturedWrites(attention_windows, joined)
    if any(pool.deferred is not None for pool in pools):
        raise RuntimeError('nested writeback capture is unsupported')
    for pool in pools:
        pool.deferred = tasks
    token = _captured.set(tasks)
    try:
        yield tasks
        if attention_windows and tasks.layers and not tasks.window_count:
            raise RuntimeError("no attention release hooks executed during graph capture")
    finally:
        _captured.reset(token)
        tasks.release_window(len(tasks.pending))
        if joined and tasks:
            torch.cuda.current_stream().wait_stream(tasks[0].pool.writeback_stream)
        for pool in pools:
            pool.deferred = None


class WritebackGraph:
    def __init__(self, forward, tasks, joined=False):
        self.forward = forward
        self.tasks = tasks  # Own captured inputs and their views across replays.
        self.stream = tasks[0].pool.writeback_stream
        self.pools = tuple(dict.fromkeys(task.pool for task in tasks))
        if any(pool.writeback_stream != self.stream for pool in self.pools):
            raise ValueError('captured layers must share one writeback stream')
        self.completion = torch.cuda.Event(external=True)
        self.background = None
        if joined:
            return
        self.background = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.background, stream=self.stream):
            for task in tasks:
                task.ready.wait()
                task.pool._finish_update(*task.tensors)
                if task.final:
                    task.pool.graph_ready.record()
        for task in tasks:
            for tensor in task.tensors:
                tensor.record_stream(self.stream)

    def replay(self):
        # A joined graph finishes every write. One shared completion dependency
        # also handles transitions from paired graphs or eager slot operations.
        seen = set()
        for pool in self.pools:
            if pool.pending and pool.completion not in seen:
                torch.cuda.current_stream().wait_event(pool.completion)
                seen.add(pool.completion)
        self.forward.replay()
        with torch.cuda.stream(self.stream if self.background is not None else torch.cuda.current_stream()):
            if self.background is not None:
                self.background.replay()
            # An ordinary submitted event also protects eager consumers/slot operations.
            self.completion.record()
            if self.background is None:
                for pool in self.pools:
                    if pool.captured_consumers:
                        pool.graph_ready.record()
        for pool in self.pools:
            pool.completion = self.completion
            pool.pending = True
