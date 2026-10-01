"""Equations (1), (7), (10), and (13); inputs are post-convolution/normalization."""
import torch


def delta_step(state, q, k, v, log_decay, beta):
    """FP32 update and readout BEFORE writeback. log_decay: [B,H] or [B,H,K]."""
    state, q, k, v, beta = (x.float() for x in (state, q, k, v, beta))
    decay = log_decay.float().exp()
    if decay.ndim == k.ndim - 1:
        decay = decay.unsqueeze(-1)
    retained = state * decay.unsqueeze(-1)
    residual = v - torch.einsum("bhk,bhkv->bhv", k, retained)
    updated = retained + k.unsqueeze(-1) * (beta.unsqueeze(-1) * residual).unsqueeze(-2)
    output = torch.einsum("bhk,bhkv->bhv", q, updated)
    return output, updated


def row_impact(q, k, log_decay, beta):
    """Squared transported read: [D(q - beta k (k^T q))]^2."""
    q, k = q.float(), k.float()
    decay = log_decay.float().exp()
    if decay.ndim == q.ndim - 1:
        decay = decay.unsqueeze(-1)
    return (decay * (q - beta.float().unsqueeze(-1) * k * (k * q).sum(-1, keepdim=True))).square()


def impact_factors(omega, gamma=0.25, floor=1e-20):
    log_w = omega.double().clamp_min(floor).log() * (gamma / 2)
    return (log_w - log_w.mean(-1, keepdim=True)).exp().float()


def lifetime_weight(mean_log_decay, horizon=2048):
    """Stable finite geometric sum, including retention exactly one or zero."""
    if horizon < 1 or torch.isnan(mean_log_decay).any() or (mean_log_decay > 0).any():
        raise ValueError("horizon must be positive and mean log retention must be <= 0")
    x = 2 * mean_log_decay.double()
    safe = torch.where(x == 0, torch.full_like(x, -1), x)
    result = torch.expm1(horizon * safe) / torch.expm1(safe)
    return torch.where(x == 0, torch.full_like(result, horizon), result)
