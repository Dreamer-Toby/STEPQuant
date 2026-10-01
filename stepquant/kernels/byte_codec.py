"""Decode signed byte carriers with independent FP16 pivot rows and scales."""
import triton as tr
import triton.language as tl


@tr.jit
def byte_decode(Page,Bits,Offsets,Rows,Columns,Slots,State,
                PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
                GROUP:tl.constexpr,KDA:tl.constexpr):
    batch,head=tl.program_id(0),tl.program_id(1)
    slot=tl.load(Slots+batch)
    i,j=tl.arange(0,K),tl.arange(0,V)
    bits=tl.load(Bits+head*K+i)
    base=Page+tl.maximum(slot,0)*PAGE
    half=base.to(tl.pointer_type(tl.float16))
    code_base=tl.load(Offsets+H*K+head)
    integer=(bits[:,None]!=16)&(slot>0)
    z=tl.load(base.to(tl.pointer_type(tl.int8))+code_base+i[:,None]+j[None,:]*K,integer,0).to(tl.float32)
    ro=tl.load(Rows+head*K+i)
    group=tl.full((K,V),0,tl.int32)
    if not KDA:group=tl.where(bits[:,None]==2,j[None,:]//GROUP,0)
    r=tl.load(half+ro[:,None]//2+group,integer,0).to(tl.float32)
    co=tl.load(Columns+head)
    c=tl.load(half+co//2+j,(co>=0)&(slot>0),0).to(tl.float32)
    off=tl.load(Offsets+head*K+i)
    pivot=tl.load(half+off[:,None]//2+j[None,:],(bits[:,None]==16)&(slot>0),0).to(tl.float32)
    x=tl.where(bits[:,None]==16,pivot,r*c[None,:]*z)
    tl.store(State+((batch*H+head)*K+i[:,None])*V+j[None,:],x)
