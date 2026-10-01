"""STEPQuant@6 background profiles; arithmetic is shared with STEPQuant@4.

Select background SM budgets by architecture and capture batch.
Explicit launch overrides remain supported.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class WritebackProfile:
    sms: int
    chunk: int
    foreground_priority: bool = True


def stepquant6_profile(architecture, batch):
    if architecture not in ('gdn', 'kda'):
        raise ValueError('architecture must be gdn or kda')
    if not isinstance(batch, int) or isinstance(batch, bool) or not 1 <= batch <= 512:
        raise ValueError('batch must be an integer in 1..512')
    tier = next(n for n in (32, 64, 128, 256, 512) if batch <= n)
    if architecture == 'gdn':
        sms = {32: 32, 64: 24, 128: 64, 256: 80, 512: 80}[tier]
        chunk = {32: 64, 64: 64, 128: 128, 256: 256, 512: 64}[tier]
    else:
        sms = 48 if tier == 512 else 32
        chunk = {32: 64, 64: 64, 128: 128, 256: 256, 512: 128}[tier]
    return WritebackProfile(sms, chunk)
