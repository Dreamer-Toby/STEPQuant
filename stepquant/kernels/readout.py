"""Prepared-input packed readout with explicit key-axis warp reductions."""
from .packed_read import load_four, load_words
from triton.experimental import gluon as tr
from triton.experimental.gluon import language as tl


@tr.jit
def readout(Page,Bits,Offsets,Rows,Columns,Slots,Q,Key,Val,Decay,Beta,Out,Delta,Partial,
            PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,GROUP:tl.constexpr,
            KDA:tl.constexpr,TILE:tl.constexpr,VS:tl.constexpr,
            HeadIds=None,WIDTH:tl.constexpr=0,NW:tl.constexpr=4,INTEGER_HEADS:tl.constexpr=False,BYTE:tl.constexpr=False,StableSlots=None,RawKey=None,RawDecay=None,ALog=None,Bias=None,HK:tl.constexpr=0,QS:tl.constexpr=0,KS:tl.constexpr=0,NORMALIZE:tl.constexpr=False,SCALE:tl.constexpr=1.):
    layout:tl.constexpr=tl.BlockedLayout([4,1],[32 if KDA else 8,1 if KDA else 4],[1,NW],[0,1]) if BYTE and WIDTH!=16 else tl.BlockedLayout([1,4],[32,1],[1,NW],[0,1]) if KDA or (RawKey is not None and WIDTH!=16) else tl.BlockedLayout([1,2],[4,8],[NW,1],[1,0])
    if KDA and not BYTE:
        batch,head,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    else:
        batch,head,tile=tl.program_id(2),tl.program_id(1),tl.program_id(0)
    if HeadIds is not None:
        head=tl.load(HeadIds+head)
    i=tl.arange(0,K,layout=tl.SliceLayout(1,layout))
    j=tile*TILE+tl.arange(0,TILE,layout=tl.SliceLayout(0,layout))
    idx=(batch*H+head)*K+i
    if RawKey is not None:
        q=tl.load(Q+batch*QS+head//(H//HK)*K+i).to(tl.float32)
        k=tl.load(Key+batch*KS+head//(H//HK)*K+i).to(tl.float32)
        if NORMALIZE:
            q*=tl.rsqrt(tl.sum(q*q,0)+1e-6)
            k*=tl.rsqrt(tl.sum(k*k,0)+1e-6)
        q*=SCALE
        g=tl.load(Decay+idx if KDA else Decay+batch*H+head).to(tl.float32)
        beta=tl.load(Beta+batch*H+head).to(tl.float32)
        if ALog is not None:
            bias=tl.load(Bias+head*K+i if KDA else Bias+head).to(tl.float32)
            u=g+bias
            g=-tl.exp(tl.load(ALog+head).to(tl.float32))*tl.where(u>20.,u,tl.log(1.+tl.exp(tl.minimum(u,20.))))
            beta=1./(1.+tl.exp(-beta))
        decay=tl.full((K,),1.,tl.float32,tl.SliceLayout(1,layout))*tl.exp(g)
        if tile==0:
            tl.store(RawKey+idx,k)
            tl.store(RawDecay+idx,decay)
    else:
        decay=tl.load(Decay+idx)
    slot=tl.load(Slots+batch)
    if StableSlots is not None:
        if (head==0)&(tile==0):tl.store(StableSlots+batch,slot)
    base=Page+tl.maximum(slot,0)*PAGE
    if BYTE:
        offset=tl.load(Offsets+head*K+i)
        stride=tl.multiple_of(tl.load(Offsets+H*K+head),4)
    elif WIDTH or (not KDA and INTEGER_HEADS):
        precision=WIDTH if WIDTH else tl.load(Bits+head*K)
        first=tl.multiple_of(tl.load(Offsets+head*K),2)
        offset=first+i*(min(4 if KDA else 16,V)*precision//8)
        stride=K*(min(4 if KDA else 16,V)*precision//8)
    else:
        offset=tl.load(Offsets+head*K+i)
        stride=tl.load(Offsets+H*K+head)
    packed=offset[:,None]+j[None,:]//min(4 if KDA else 16,V)*stride
    pivot_ptr=(offset[:,None]//2+j[None,:]) if BYTE else (packed//2+j[None,:]%min(4 if KDA else 16,V))
    half=base.to(tl.pointer_type(tl.float16))
    if WIDTH==16:
        x=tl.load(half+pivot_ptr,slot>0,0).to(tl.float32)*decay[:,None]
    else:
        if WIDTH:
            bits=tl.full((K,),WIDTH,tl.int32,tl.SliceLayout(1,layout))
        else:
            bits=tl.load(Bits+head*K+i) if KDA else tl.full((K,),tl.load(Bits+head*K),tl.int32,tl.SliceLayout(1,layout))
        width=tl.minimum(bits,8)
        integer=(bits[:,None]!=16)&(slot>0)
        if BYTE:
            z=tl.load(base.to(tl.pointer_type(tl.int8))+stride+i[:,None]+j[None,:]*K,(slot>0)&(tl.load(Columns+head)>=0),0,cache_modifier=".cg" if KDA else "").to(tl.float32)
        else:
            shift=(j[None,:]*width[:,None])%8
            addr=packed+(j[None,:]%min(4 if KDA else 16,V))*width[:,None]//8
            integer=(bits[:,None]!=16)&(slot>0)
            if KDA:
                code=load_four(base,offset,bits,stride,slot,tile,K,TILE,NW,layout)
            elif K>=4 and V>=16:
                code=load_words(base,offset,stride,slot,tile,WIDTH if WIDTH else tl.load(Bits+head*K),K,TILE,NW,layout,KEY_LANES=32 if RawKey is not None else 4,KEY_WARPS=1 if RawKey is not None else 0)
            else:
                lo=tl.load(base+addr,integer,0).to(tl.int32)
                hi=tl.load(base+addr+1,integer&(shift+width[:,None]>8),0).to(tl.int32)
                code=((lo|(hi<<8))>>shift)&((1<<width[:,None])-1)

            z=(code-((1<<(width[:,None]-1))-1)).to(tl.float32)
            if not KDA:
                z=tl.where(bits[:,None]==2,2.*code-3.,z)
        row=tl.load(Rows+head*K+i)
        group=tl.full((K,TILE),0,tl.int32,layout)
        if not KDA:
            group=tl.where(bits[:,None]==2,j[None,:]//GROUP,0)
        if KDA or GROUP>=TILE:
            rg=tl.full((K,),0,tl.int32,tl.SliceLayout(1,layout))
            if not KDA:rg=tl.where(bits==2,tile*TILE//GROUP,0)
            r=tl.load(half+row//2+rg,(bits!=16)&(slot>0),0).to(tl.float32)
            if KDA and not BYTE:r*=(1<<(8-width)).to(tl.float32)
            r=(r*decay)[:,None]
        else:
            r=tl.load(half+row[:,None]//2+group,integer,0).to(tl.float32)*decay[:,None]
        col=tl.load(Columns+head)
        c=tl.load(half+col//2+j,(col>=0)&(slot>0),0).to(tl.float32)
        x=r*c[None,:]*z
        if KDA or (WIDTH==0 and not INTEGER_HEADS):
            pivot=tl.load(half+pivot_ptr,(bits[:,None]==16)&(slot>0),0).to(tl.float32)
            x=tl.where(bits[:,None]==16,pivot*decay[:,None],x)
    idx=(batch*H+head)*K+i
    if RawKey is None:
        k=tl.load(Key+idx)
        q=tl.load(Q+idx)
        beta=tl.load(Beta+batch*H+head)
    value=tl.load(Val+batch*VS+head*V+j).to(tl.float32)
    delta=beta*(value-tl.sum(k[:,None]*x,0))
    x+=k[:,None]*delta[None,:]
    output=tl.sum(q[:,None]*x,0)
    tl.store(Out+(batch*H+head)*V+j,tl.where(slot>0,output,0.))
    tl.store(Delta+(batch*H+head)*V+j,delta)
    if WIDTH!=16:
        tl.store(Partial+((batch*H+head)*(V//TILE)+tile)*K+i,tl.sum(tl.abs(x),1))
    if WIDTH==16:
        tl.store(half+pivot_ptr,x,slot>0)
    elif KDA or (WIDTH==0 and not INTEGER_HEADS):
        tl.store(half+pivot_ptr,x,(bits[:,None]==16)&(slot>0))
