"""Aligned four-code stores; INT6 uses exactly three bytes, never a padded byte."""
import triton as tr
import triton.language as tl


@tr.jit
def store_words(base,offset,bits,codes,start,K:tl.constexpr,V:tl.constexpr,Stride=None):
    lane=tl.arange(0,4)
    group=tl.arange(0,V//4)
    width=tl.minimum(bits,8)
    grouped=tl.reshape(codes,(K,V//4,4))
    word=tl.sum(grouped<<(lane[None,None,:]*width[:,None,None]),2)
    if Stride is not None:
        byte=(word[:,:,None]>>(lane[None,None,:]*8)).to(tl.uint8)
        addr=offset[:,None,None]+group[None,:,None]*Stride[:,None,None]+lane[None,None,:]
        tl.store(base+addr,byte,(bits[:,None,None]!=16)&(bits[:,None,None]>0)&(lane[None,None,:]<width[:,None,None]//2))
    elif K>=4 and V>=16:
        # Pages are half-aligned, including mixed KDA maps with odd integer-row counts.
        tl.store(base+offset[:,None]+start//4+group[None,:],word.to(tl.uint8),bits[:,None]==2)
        tl.store(base.to(tl.pointer_type(tl.uint16))+offset[:,None]//2+start//4+group[None,:],word.to(tl.uint16),bits[:,None]==4)
        half=tl.arange(0,2)
        ptr=base.to(tl.pointer_type(tl.uint16))+offset[:,None,None]//2+(start//4+group[None,:,None])*2+half[None,None,:]
        tl.store(ptr,(word[:,:,None]>>(16*half[None,None,:])).to(tl.uint16),bits[:,None,None]==8)
        byte=(word[:,:,None]>>(lane[None,None,:]*8)).to(tl.uint8)
        addr=offset[:,None,None]+(start//4+group[None,:,None])*3+lane[None,None,:]
        tl.store(base+addr,byte,(bits[:,None,None]==6)&(lane[None,None,:]<3))
    else:
        byte=(word[:,:,None]>>(lane[None,None,:]*8)).to(tl.uint8)
        addr=offset[:,None,None]+(start//4+group[None,:,None])*(width[:,None,None]//2)+lane[None,None,:]
        tl.store(base+addr,byte,(bits[:,None,None]!=16)&(bits[:,None,None]>0)&(lane[None,None,:]<width[:,None,None]//2))
