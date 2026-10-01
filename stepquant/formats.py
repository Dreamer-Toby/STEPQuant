"""Explicit recurrent-state formats and independent PyTorch numerical oracles.

All matrices use [batch, head, key, value]. Groupwise formats group value
coordinates.
DSQ adapts Q-Mamba's state codec only, without ESR or weight quantization.
"""
from dataclasses import dataclass, asdict
import re
import torch


@dataclass(frozen=True)
class StateFormat:
    name: str
    kind: str
    bits: int = 8
    group_size: int = 32
    group_axis: str = 'v'

    @property
    def calibrated(self):
        return self.kind == 'stepquant'

    def to_dict(self):
        return asdict(self)


def get_format(name):
    if name in ('stepquant', 'stepquant4', 'stepquant6'):
        return StateFormat(name, 'stepquant', 4 if name.endswith('4') else 6, 32)
    if name == 'fp32':
        return StateFormat(name, 'fp32', 32, 32)
    match = re.fullmatch(r'sym_int([468])_g128', name)
    if match:
        return StateFormat(name, 'symmetric', int(match[1]), 128)
    match = re.fullmatch(r'dsq_int([46])', name)
    if match:
        return StateFormat(name, 'dsq', int(match[1]), 128, group_axis='k')
    raise ValueError(f'unknown state format: {name}')


BASELINE_NAMES = ['fp32', 'sym_int4_g128', 'sym_int6_g128', 'sym_int8_g128',
                  'dsq_int4', 'dsq_int6']
FORMAT_NAMES = BASELINE_NAMES + ['stepquant4', 'stepquant6']


def half_scale(x):
    return x.clamp(2**-24, 65504).half().float()


def group_fit(x, spec):
    """Return codes, stored scale, stored zero point, reconstruction (FP32)."""
    x = x.float()
    if spec.kind == 'fp32':
        return x, None, None, x
    if spec.kind != 'symmetric':
        raise ValueError('group_fit requires FP32 or symmetric rowwise absmax')
    g = spec.group_size
    if x.shape[-1] % g:
        raise ValueError('value dimension must be divisible by group size')
    y = x.reshape(*x.shape[:-1], -1, g)
    limit = 2**(spec.bits - 1) - 1
    s = half_scale(y.abs().amax(-1, keepdim=True) / limit)
    z = torch.full_like(s, limit)
    q = (y / s).round().clamp(-limit, limit) + z
    return q.reshape(x.shape), s, z, ((q - z) * s).reshape(x.shape)


def dual_fit(x, bits, impact):
    """Q-Mamba DSQ symmetric dual-axis reconstruction (no weighted fit)."""
    x = x.float()
    if x.shape[-2] > 128:
        raise ValueError('dual-axis g128 currently requires key_dim <= 128')
    b = bits.to(x.device)[None, :, :, None]
    mask = b != 16
    r = half_scale(x.abs().mean(-1, keepdim=True).clamp_min(1e-20).sqrt())
    y = x / r
    levels = 2**(b.clamp_max(8) - 1) - 1
    c = half_scale(torch.where(mask, y.abs() / levels, 0).amax(-2, keepdim=True))
    q = (y / c).round().maximum(-levels).minimum(levels)
    recon = torch.where(mask, r * c * q, x.half().float())
    return q + levels, r, c, torch.zeros_like(c), recon


@dataclass
class FormatPlan:
    architecture: str
    bits: torch.Tensor
    impact: torch.Tensor
    value_group_size: int = 32

    def __post_init__(self):
        if self.architecture not in ('gdn', 'kda') or self.bits.ndim != 2 or self.bits.shape != self.impact.shape:
            raise ValueError('invalid format plan shape/architecture')
        if not torch.isin(self.bits, torch.tensor([2,4,6,8,16,32],device=self.bits.device)).all():
            raise ValueError('invalid state precision')
        if not torch.isfinite(self.impact).all() or (self.impact <= 0).any():
            raise ValueError('impact must be finite and positive')

    def to(self, device):
        return FormatPlan(self.architecture,self.bits.to(device),self.impact.to(device),self.value_group_size)
