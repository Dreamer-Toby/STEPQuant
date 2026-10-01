"""Coalesced row statistics and affine coefficients; value sums stay local."""
from triton.experimental import gluon as tr
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language.extra import libdevice


@tr.jit
def _row_coefficients(Page,Bits,Impact,RowOffsets,Slots,UpdateKey,Decay,Partial,Coefficients,
                        PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,GROUP:tl.constexpr,
                        KDA:tl.constexpr,ROWTILE:tl.constexpr,NW:tl.constexpr=4,GROUPED:tl.constexpr=False):
    head,batch=tl.program_id(0),tl.program_id(1)
    L:tl.constexpr=tl.BlockedLayout([1,1],[32,1],[NW,1],[0,1])
    i=tl.arange(0,K,layout=tl.SliceLayout(1,L))
    t=tl.arange(0,V//ROWTILE,layout=tl.SliceLayout(0,L))
    slot=tl.load(Slots+batch)
    if KDA:
        b=tl.load(Bits+head*K+i)
        integer=b!=16
    else:
        integer=tl.full((K,),True,tl.int1,tl.SliceLayout(1,L))
    w=tl.load(Impact+head*K+i)
    roff=tl.load(RowOffsets+head*K+i)
    sums=tl.load(Partial+((batch*H+head)*(V//ROWTILE)+t[None,:])*K+i[:,None])
    mean=tl.sum(sums,1)/V
    NG:tl.constexpr=V//GROUP if GROUPED else 1
    groups=tl.arange(0,NG,layout=tl.SliceLayout(0,L))
    means=mean[:,None]+tl.full((K,NG),0.,tl.float32,L)
    if GROUPED:
        grouped=tl.reshape(sums,(K,V//GROUP,GROUP//ROWTILE))
        means=tl.convert_layout(tl.sum(grouped,2),L)/GROUP
    r=tl.sqrt(tl.maximum(means,1e-20)/w[:,None])
    r=tl.minimum(tl.maximum(r,2.**-24),65504.).to(tl.float16).to(tl.float32)
    rg=tl.full((K,NG),0,tl.int32,L)
    if GROUPED:rg=groups[None,:]+tl.full((K,NG),0,tl.int32,L)
    base=(Page+tl.maximum(slot,0)*PAGE).to(tl.pointer_type(tl.float16))
    old=tl.load(base+roff[:,None]//2+rg,integer[:,None]&(slot>0),0).to(tl.float32)
    decay=tl.load(Decay+(batch*H+head)*K+i)
    norm=tl.full((K,),1.,tl.float32,tl.SliceLayout(1,L))
    if KDA:norm=(1<<(8-tl.minimum(b,8))).to(tl.float32)
    rn=r*norm[:,None]
    metric=tl.where(integer[:,None],w[:,None]*w[:,None]*rn*rn,0.)
    stride=H*K*(V//GROUP)
    idx=batch*3*stride+head*K*(V//GROUP)+groups[None,:]*K+i[:,None]
    inverse=libdevice.fast_dividef(1.,rn)
    key=tl.load(UpdateKey+(batch*H+head)*K+i)
    tl.store(Coefficients+idx,key[:,None]*inverse)
    tl.store(Coefficients+stride+idx,(old*decay[:,None])*inverse)
    tl.store(Coefficients+2*stride+idx,metric)
    tl.store(base+roff[:,None]//2+rg,r,integer[:,None]&(slot>0))


@tr.jit
def prepare_rows_kernel(Page,Bits,Impact,RowOffsets,Slots,UpdateKey,Decay,Partial,Coefficients,
                        PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,GROUP:tl.constexpr,
                        KDA:tl.constexpr,ROWTILE:tl.constexpr,NW:tl.constexpr=4):
    if KDA:
        _row_coefficients(Page,Bits,Impact,RowOffsets,Slots,UpdateKey,Decay,Partial,Coefficients,PAGE,H,K,V,GROUP,KDA,ROWTILE,NW,False)
    else:
        bit=tl.load(Bits+tl.program_id(0)*K)
        if bit==2:
            _row_coefficients(Page,Bits,Impact,RowOffsets,Slots,UpdateKey,Decay,Partial,Coefficients,PAGE,H,K,V,GROUP,KDA,ROWTILE,NW,True)
        elif bit!=16:
            _row_coefficients(Page,Bits,Impact,RowOffsets,Slots,UpdateKey,Decay,Partial,Coefficients,PAGE,H,K,V,GROUP,KDA,ROWTILE,NW,False)
