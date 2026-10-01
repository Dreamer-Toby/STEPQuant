"""Portable little-endian bit packing. No persistent int8 surrogate for INT2/4/6."""
import math
import torch


def pack(codes, bits):
    if bits not in (2, 4, 6, 8):
        raise ValueError("supported code widths: 2, 4, 6, 8")
    flat = codes.reshape(-1).to(torch.int32)
    n = 8 // math.gcd(bits, 8)
    width = n * bits // 8
    flat = torch.nn.functional.pad(flat, (0, (-flat.numel()) % n)).reshape(-1, n)
    word = (flat << (torch.arange(n, device=flat.device) * bits)).sum(-1)
    return ((word[:, None] >> (torch.arange(width, device=flat.device) * 8)) & 255).to(torch.uint8).flatten()


def unpack(data, bits, count):
    n = 8 // math.gcd(bits, 8)
    width = n * bits // 8
    word = (data.to(torch.int64).reshape(-1, width) << (torch.arange(width, device=data.device) * 8)).sum(-1)
    codes = (word[:, None] >> (torch.arange(n, device=data.device) * bits)) & ((1 << bits) - 1)
    return codes.flatten()[:count].to(torch.int16)
