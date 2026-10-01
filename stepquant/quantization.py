"""Dual-axis fitting (Eq. 11-14, App. C) and compressed persistent storage."""
from dataclasses import dataclass
import torch
from .packing import pack, unpack


@dataclass
class QuantizationPlan:
    architecture: str
    bits: torch.Tensor  # [H,K], 16 denotes pivots
    impact: torch.Tensor  # [H,K], normalized w, NOT omega
    value_group_size: int = 32  # Qwen INT2: implementation choice, not specified in PDF

    def __post_init__(self):
        if self.architecture not in ("gdn", "kda"):
            raise ValueError("architecture must be gdn or kda")
        if self.bits.ndim != 2 or self.bits.shape != self.impact.shape:
            raise ValueError("bits and impact must have shape [heads, key_dim]")
        if not torch.isin(self.bits, torch.tensor([2, 4, 6, 8, 16], device=self.bits.device)).all():
            raise ValueError("precision must be 2, 4, 6, 8 or 16")
        if self.architecture == "gdn" and not (self.bits == self.bits[:, :1]).all():
            raise ValueError("GDN allocation is per whole head")
        if not torch.isfinite(self.impact).all() or (self.impact <= 0).any():
            raise ValueError("impact factors must be finite and positive")
        if self.value_group_size < 1:
            raise ValueError("value_group_size must be positive")

    def to(self, device):
        return QuantizationPlan(self.architecture, self.bits.to(device), self.impact.to(device),
                                self.value_group_size)


@dataclass
class PackedState:
    # Layout and row indices belong to the shared codec, not to each request.
    codes: dict
    rows: torch.Tensor
    columns: torch.Tensor
    pivots: torch.Tensor
    two_bit_rows: torch.Tensor | None
    shape: tuple

    @property
    def nbytes(self):
        values = [*self.codes.values(), self.rows, self.columns, self.pivots, self.two_bit_rows]
        return sum(x.numel() * x.element_size() for x in values if x is not None)


def _fp16_scale(x):
    # Fit using the scales actually stored, avoiding a calibration/inference discrepancy.
    return x.clamp(2**-24, 65504).half().float()


def fit_state(x, plan):
    """Return integer levels, stored row/column factors and reconstruction.

    One weighted fit per update, followed by final nearest-level selection.
    KDA fits one column vector across ALL non-pivot rows at mixed precisions.
    Qwen INT2 uses signed levels {-3,-1,1,3} and per-value-group row factors.
    """
    x = x.float()
    bits = plan.bits.to(x.device)
    w = plan.impact.to(x.device)[None, :, :, None]
    integer = (bits != 16)[None, :, :, None]
    qmax = ((2.0 ** (bits.clamp_max(8).float() - 1)) - 1)[None, :, :, None]
    norm = (2.0 ** (8 - bits.clamp_max(8).float()))[None, :, :, None] if plan.architecture == "kda" else torch.ones_like(qmax)
    r = _fp16_scale((x.abs().mean(-1, keepdim=True).clamp_min(1e-20) / w).sqrt())
    r_full = r.expand_as(x).clone()
    two_rows = None
    two_heads = (bits[:, 0] == 2) if plan.architecture == "gdn" else torch.zeros(bits.shape[0], dtype=torch.bool, device=x.device)
    if two_heads.any():
        if x.shape[-1] % plan.value_group_size:
            raise ValueError("Qwen INT2 value dimension must be divisible by value_group_size")
        grouped = x[:, two_heads].reshape(*x[:, two_heads].shape[:-1], -1, plan.value_group_size)
        two_rows = _fp16_scale((grouped.abs().mean(-1).clamp_min(1e-20) / w[:, two_heads]).sqrt())
        r_full[:, two_heads] = two_rows.repeat_interleave(plan.value_group_size, -1)
        qmax[:, two_heads] = 3
    scaled = x / r_full
    c = _fp16_scale(torch.where(integer, scaled.abs() / (qmax * norm), 0).amax(-2, keepdim=True))

    def encode(columns):
        y = scaled / columns / norm
        z = torch.round(y).maximum(-qmax).minimum(qmax)
        if two_heads.any():
            y2 = y[:, two_heads]
            z[:, two_heads] = torch.where(y2 >= 0, 1., -1.) * torch.where(y2.abs() >= 2, 3., 1.)
        return z

    z = encode(c)
    v = r_full * norm * z
    weights = w.square() * integer
    num = (weights * v * x).sum(-2, keepdim=True)
    den = (weights * v.square()).sum(-2, keepdim=True)
    c = _fp16_scale(torch.where(den > 0, num / den.clamp_min(1e-30), c))
    z = encode(c)
    reconstructed = r_full * c * norm * z
    reconstructed = torch.where(integer, reconstructed, x.half().float())
    return z.to(torch.int16), r.squeeze(-1).half(), c.squeeze(-2).half(), two_rows.half() if two_rows is not None else None, reconstructed


class StateCodec:
    def __init__(self, plan):
        self.plan = plan
        self.indices = {b: (plan.bits.flatten() == b).nonzero().flatten() for b in (2, 4, 6, 8, 16)}
        self.integer_rows = (plan.bits.flatten() != 16).nonzero().flatten()
        self.integer_heads = (plan.bits != 16).any(-1).nonzero().flatten()
        self.two_heads = (plan.bits[:, 0] == 2).nonzero().flatten() if plan.architecture == "gdn" else None

    def encode(self, state):
        if tuple(state.shape[1:3]) != tuple(self.plan.bits.shape):
            raise ValueError("state/precision-map shape mismatch")
        z, r, c, r2, _ = fit_state(state, self.plan)
        batch, heads, keys, values = state.shape
        z = z.reshape(batch, heads * keys, values)
        packed = {}
        for bits in (2, 4, 6, 8):
            idx = self.indices[bits]
            codes = z[:, idx]
            if bits == 2 and self.plan.architecture == "gdn":
                codes = (codes + 3) // 2
            else:
                codes = codes + (2 ** (bits - 1) - 1)
            packed[bits] = pack(codes, bits)
        pivots = state.reshape(batch, heads * keys, values)[:, self.indices[16]].half()
        return PackedState(packed, r.reshape(batch, -1)[:, self.integer_rows],
                           c[:, self.integer_heads], pivots, r2, tuple(state.shape))

    def decode(self, packed):
        batch, heads, keys, values = packed.shape
        device = packed.rows.device
        z = torch.zeros((batch, heads * keys, values), device=device)
        for bits in (2, 4, 6, 8):
            idx = self.indices[bits]
            codes = unpack(packed.codes[bits], bits, batch * idx.numel() * values).reshape(batch, -1, values).float()
            if bits == 2 and self.plan.architecture == "gdn":
                codes = 2 * codes - 3
            else:
                codes -= 2 ** (bits - 1) - 1
            if self.plan.architecture == "kda":
                codes *= 2 ** (8 - bits)
            z[:, idx] = codes
        r = torch.ones((batch, heads * keys), device=device)
        r[:, self.integer_rows] = packed.rows.float()
        c = torch.ones((batch, heads, values), device=device)
        c[:, self.integer_heads] = packed.columns.float()
        result = z.reshape(batch, heads, keys, values) * r.reshape(batch, heads, keys, 1) * c.unsqueeze(-2)
        if packed.two_bit_rows is not None:
            result[:, self.two_heads] = (z.reshape(batch, heads, keys, values)[:, self.two_heads]
                * packed.two_bit_rows.float().repeat_interleave(self.plan.value_group_size, -1)
                * c[:, self.two_heads, None, :])
        result.reshape(batch, heads * keys, values)[:, self.indices[16]] = packed.pivots.float()
        return result
