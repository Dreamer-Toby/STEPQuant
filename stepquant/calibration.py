"""Offline calibration from post-convolution recurrent traces; no quantized feedback."""
from dataclasses import asdict
import torch
from .core import row_impact, impact_factors, lifetime_weight
from .quantization import QuantizationPlan, fit_state
from .allocation import allocate


class LayerStatistics:
    def __init__(self, architecture, sample_every=8, max_snapshots=64):
        self.architecture = architecture
        self.sample_every = sample_every
        self.max_snapshots = max_snapshots
        self.count = 0
        self.updates = 0
        self.omega = None
        self.log_decay = None
        self.snapshots = []
        self.sample_count = 0
        self.rng = torch.Generator().manual_seed(0)

    def observe(self, q, k, v, log_decay, beta, state):
        omega = row_impact(q, k, log_decay, beta).double().sum(0).cpu()
        decay = log_decay.double().sum(0).cpu()
        self.omega = omega if self.omega is None else self.omega + omega
        self.log_decay = decay if self.log_decay is None else self.log_decay + decay
        self.count += q.shape[0]
        self.updates += 1
        if self.updates % self.sample_every == 0:
            for sample in state:
                self.sample_count += 1
                if len(self.snapshots) < self.max_snapshots:
                    self.snapshots.append(sample.float().cpu().clone())
                else:
                    i = torch.randint(self.sample_count, (), generator=self.rng).item()
                    if i < self.max_snapshots:
                        self.snapshots[i] = sample.float().cpu().clone()

    def export(self):
        if not self.snapshots:
            raise ValueError("no state samples collected; increase calibration tokens or lower sample_every")
        return dict(architecture=self.architecture, omega=self.omega / self.count,
                    log_decay=self.log_decay / self.count, snapshots=torch.stack(self.snapshots),
                    tokens=self.count, sampled_states=self.sample_count)


def _distortions(stat, candidates, impact, pivots, group_size, device):
    architecture = stat['architecture']
    shape = impact.shape
    losses = []
    fp_total = 0
    weights = impact[None, :, :, None].square() if architecture == 'gdn' else None
    dims = (0, 2, 3) if architecture == 'gdn' else (0, 3)
    for bit in candidates:
        bits = torch.full(shape, bit, dtype=torch.long, device=device)
        bits[pivots] = 16
        plan = QuantizationPlan(architecture, bits, impact, group_size)
        total = 0
        for sample in stat['snapshots']:
            x = sample[None].to(device)
            reconstructed = fit_state(x, plan)[-1]
            error = (reconstructed - x).square()
            if weights is not None:
                error = error * weights
            total = total + error.mean(dims).cpu().double()
            # FP16 reference error is independent of candidate precision.
            if bit == candidates[0]:
                fp_error = (x.half().float() - x).square()
                if weights is not None:
                    fp_error = fp_error * weights
                fp_total = fp_total + fp_error.mean(dims).cpu().double()
        losses.append(total / len(stat['snapshots']))
    return torch.stack(losses, -1), fp_total / len(stat['snapshots'])


def calibrate(statistics, nominal_bits=6, pivots=None, horizon=2048,
              value_group_size=32, device='cpu'):
    """Global allocation across layers. Nominal budget counts each FP16 pivot as INT8.

    Qwen: initial DP -> rank allocated residual risk -> pivots -> residual DP.
    KDA: rank lifetime-weighted INT8-to-FP16 gain -> exclude pivots -> refit -> allocate.
    """
    if nominal_bits not in (4, 6) or not statistics:
        raise ValueError("provide statistics and nominal_bits=4 or 6")
    names = sorted(statistics)
    architecture = statistics[names[0]]['architecture']
    if any(s['architecture'] != architecture for s in statistics.values()):
        raise ValueError("one architecture per calibration artifact")
    candidates = [2, 4, 6, 8] if nominal_bits == 4 else [4, 6, 8]
    info, all_cost, all_fp = {}, [], []
    for name in names:
        s = statistics[name]
        w = impact_factors(s['omega']).to(device)
        mask = torch.zeros_like(w, dtype=torch.bool)
        d, fp = _distortions(s, candidates, w, mask, value_group_size, device)
        lifetime = lifetime_weight(s['log_decay'], horizon)
        cost = (d * lifetime[..., None]).reshape(-1, len(candidates))
        info[name] = (w, lifetime, d.shape[:-1])
        all_cost.append(cost)
        all_fp.append((fp * lifetime).flatten())
    cost = torch.cat(all_cost)
    fp = torch.cat(all_fp)
    n = len(cost)
    pivots = (32 if architecture == 'gdn' else 512) if pivots is None else pivots
    if not 0 <= pivots <= n:
        raise ValueError(f"pivot count {pivots} exceeds {n} units; set --pivots for small models")
    integer_budget = nominal_bits * n - 8 * pivots
    if integer_budget < (n - pivots) * min(candidates):
        raise ValueError("too many pivots for nominal pre-pivot budget")
    if architecture == 'gdn':
        initial = allocate(cost, candidates, nominal_bits * n, architecture)
        choice = torch.searchsorted(torch.tensor(candidates), initial)
        risk = cost.gather(1, choice[:, None]).flatten()
    else:
        risk = cost[:, -1] - fp
    pivot_mask = torch.zeros(n, dtype=torch.bool)
    pivot_mask[torch.argsort(risk, descending=True, stable=True)[:pivots]] = True
    if architecture == 'kda' and pivots:
        offset, revised = 0, []
        for name in names:
            w, lifetime, unit_shape = info[name]
            size = lifetime.numel()
            mask = pivot_mask[offset:offset + size].reshape(w.shape).to(device)
            d, _ = _distortions(statistics[name], candidates, w, mask, value_group_size, device)
            revised.append((d * lifetime[..., None]).reshape(-1, len(candidates)))
            offset += size
        cost = torch.cat(revised)
    assignment = torch.full((n,), 16, dtype=torch.long)
    assignment[~pivot_mask] = allocate(cost[~pivot_mask], candidates, integer_budget, architecture)
    plans, offset = {}, 0
    for name in names:
        w, lifetime, unit_shape = info[name]
        size = lifetime.numel()
        bits = assignment[offset:offset + size].reshape(unit_shape)
        if architecture == 'gdn':
            bits = bits[:, None].expand(w.shape).clone()
        plans[name] = asdict(QuantizationPlan(architecture, bits.cpu(), w.cpu(), value_group_size))
        offset += size
    return dict(format_version=2, plans=plans, settings=dict(nominal_bits=nominal_bits,
                pivots=pivots, horizon=horizon, value_group_size=value_group_size,
                candidates=candidates, integer_budget=integer_budget,
                integer_bits_used=int(assignment[~pivot_mask].sum()),
                calibration_tokens={name: statistics[name]['tokens'] for name in names}))
