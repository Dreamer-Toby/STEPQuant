"""Frozen candidate distortion allocation: exact head DP and row Lagrangian repair."""
import heapq
import numpy as np
import torch


def allocate_dp(cost, candidates, budget):
    cost = np.asarray(cost, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=int)
    n, choices = cost.shape
    if budget < n * candidates.min():
        raise ValueError("integer budget is infeasible")
    budget = min(int(budget), n * int(candidates.max()))
    dp = np.full(budget + 1, np.inf)
    dp[0] = 0
    back = np.full((n, budget + 1), -1, dtype=np.int8)
    for i in range(n):
        nxt = np.full_like(dp, np.inf)
        for j, b in enumerate(candidates):
            trial = dp[:budget + 1 - b] + cost[i, j]
            better = trial < nxt[b:]
            nxt[b:][better] = trial[better]
            back[i, b:][better] = j
        dp = nxt
    remaining = int(dp.argmin())
    result = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        j = back[i, remaining]
        if j < 0:
            raise RuntimeError("allocation backtracking failed")
        result[i] = candidates[j]
        remaining -= candidates[j]
    return torch.from_numpy(result)


def allocate_lagrangian(cost, candidates, budget):
    """Feasible Lagrangian solution with gain-per-bit discrete repair (not exact DP)."""
    cost = np.asarray(cost, dtype=np.float64)
    b = np.asarray(candidates, dtype=int)
    n = len(cost)
    if budget < n * b.min():
        raise ValueError("integer budget is infeasible")
    if n == 0:
        return torch.empty(0, dtype=torch.long)
    choose = lambda lam: np.argmin(cost + lam * b, axis=1)
    pick = choose(0.)
    if b[pick].sum() <= budget:
        return torch.from_numpy(b[pick].copy())
    lo, hi = 0., max(float(np.ptp(cost, axis=1).max()), 1e-30)
    for _ in range(80):
        mid = (lo + hi) / 2
        trial = choose(mid)
        if b[trial].sum() > budget:
            lo = mid
        else:
            hi, pick = mid, trial
    remaining = int(budget - b[pick].sum())
    # Consider every improving upgrade, including non-monotone candidate curves.
    heap = []
    def enqueue(i):
        old = pick[i]
        for new in range(old + 1, len(b)):
            gain = cost[i, old] - cost[i, new]
            if gain > 0:
                heapq.heappush(heap, (-gain / (b[new] - b[old]), i, old, new))
    for i in range(n):
        enqueue(i)
    while heap and remaining:
        _, i, old, new = heapq.heappop(heap)
        if old != pick[i] or b[new] - b[old] > remaining:
            continue
        remaining -= int(b[new] - b[old])
        pick[i] = new
        enqueue(i)
    return torch.from_numpy(b[pick].copy())


def allocate(cost, candidates, budget, architecture):
    if not np.isfinite(np.asarray(cost)).all():
        raise ValueError("candidate distortion must be finite")
    return (allocate_dp if architecture == "gdn" else allocate_lagrangian)(cost, candidates, budget)
