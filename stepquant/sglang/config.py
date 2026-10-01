"""Shared request limits and CUDA Graph sizes for serving and evaluation."""
BATCH_SIZES = (32, 64, 128, 256, 512)


def request_limit(workload, override=None):
    defaults = {'long': 64, 'short': 256}
    limit = defaults[workload] if override is None else override
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 512:
        raise ValueError('max-running-requests must be an integer in 1..512')
    return limit


def graph_batches(limit):
    request_limit('long', limit)
    return sorted({n for n in (1, 2, 4, 8, 16, *BATCH_SIZES, limit) if n <= limit})
