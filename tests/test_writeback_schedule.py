"""A release schedule must not accumulate work across recurrent layers."""
from types import SimpleNamespace
import pytest
import torch
from stepquant.kernels.writeback import CapturedWrites


@pytest.mark.parametrize('chunk',[32,64,128,256,512])
def test_kda_release_keeps_forward_tail_bounded(monkeypatch,chunk):
    monkeypatch.setattr(torch.cuda,'Event',lambda **kwargs:SimpleNamespace(record=lambda:None))
    tasks=CapturedWrites(attention_windows=True)
    slots=torch.arange(512)
    recurrent=[True,True,True,False]*6+[True,True,False]
    chunks=(len(slots)+chunk-1)//chunk
    for is_recurrent in recurrent:
        tasks.release_window()
        if is_recurrent:
            pool=SimpleNamespace(plan=SimpleNamespace(architecture='kda'),storage='byte',writeback_chunk=chunk)
            tasks.append((pool,slots))
        assert len(tasks.pending)<=chunks, 'writeback work accumulated across layers'
    assert not tasks.pending, 'the final full-attention window should drain the pending chunks'
    assert len(tasks)==sum(recurrent)*chunks
    assert sum(task.final for task in tasks)==sum(recurrent)
