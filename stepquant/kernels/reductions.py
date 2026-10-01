"""Joint reductions share cross-warp synchronization between fitting statistics."""
import triton as tr
import triton.language as tl


@tr.jit
def sum_pair(a,b,c,d):
    return a+c,b+d


@tr.jit
def sum_triple(a,b,c,d,e,f):
    return a+d,b+e,c+f


@tr.jit
def minmax(a,b,c,d):
    return tl.minimum(a,c),tl.maximum(b,d)
