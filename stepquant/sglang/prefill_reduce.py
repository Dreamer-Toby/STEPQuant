"""Accumulate prefill TP sums in FP32 before returning the model dtype.

BF16 reductions round after intermediate additions. Different tensor
partitions can use different rank orders, splitting identical prompt rows
before recurrent-state quantization. Decode keeps its existing collective.
"""
from contextvars import ContextVar
import torch

_prefill = ContextVar('stepquant_prefill_reduce', default=False)


def forward(original, self, forward_batch, *args, **kwargs):
    token = _prefill.set(forward_batch.forward_mode.is_extend())
    try:
        return original(self, forward_batch, *args, **kwargs)
    finally:
        _prefill.reset(token)


def all_reduce(original, self, input_, *args, **kwargs):
    if (_prefill.get() and self.world_size > 1
            and input_.dtype in (torch.bfloat16, torch.float16)):
        return original(self, input_.float(), *args, **kwargs).to(input_.dtype)
    return original(self, input_, *args, **kwargs)
