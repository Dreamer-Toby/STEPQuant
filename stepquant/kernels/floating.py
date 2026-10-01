"""Typed, contiguous state paths for formats that do not need integer metadata."""
import triton as tr
import triton.language as tl


@tr.jit
def codec(Page,Slots,State,N:tl.constexpr,ENCODE:tl.constexpr,BLOCK:tl.constexpr=1024):
    batch,tile=tl.program_id(0),tl.program_id(1)
    slot=tl.load(Slots+batch)
    i=tile*BLOCK+tl.arange(0,BLOCK)
    if ENCODE:
        x=tl.load(State+batch*N+i,i<N,0)
        tl.store(Page+tl.maximum(slot,0)*N+i,x,(i<N)&(slot>0))
    else:
        x=tl.load(Page+tl.maximum(slot,0)*N+i,(i<N)&(slot>0),0)
        tl.store(State+batch*N+i,x,i<N)


@tr.jit
def decode(Page,Slots,Q,Key,Val,G,Beta,Out,ALog,Bias,
           H:tl.constexpr,HK:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
           KDA:tl.constexpr,NORMALIZE:tl.constexpr,RAW_GATES:tl.constexpr,
           SCALE:tl.constexpr,QS:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr,TILE:tl.constexpr=32):
    batch,head,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    slot=tl.load(Slots+batch)
    i,j=tl.arange(0,K),tile*TILE+tl.arange(0,TILE)
    q=tl.load(Q+batch*QS+head//(H//HK)*K+i).to(tl.float32)
    k=tl.load(Key+batch*KS+head//(H//HK)*K+i).to(tl.float32)
    v=tl.load(Val+batch*VS+head*V+j).to(tl.float32)
    beta=tl.load(Beta+batch*H+head).to(tl.float32)
    if KDA:
        g=tl.load(G+(batch*H+head)*K+i).to(tl.float32)
    else:
        g=tl.load(G+batch*H+head).to(tl.float32)
    if RAW_GATES:
        if KDA:bias=tl.load(Bias+head*K+i).to(tl.float32)
        else:bias=tl.load(Bias+head).to(tl.float32)
        u=g+bias
        g=-tl.exp(tl.load(ALog+head).to(tl.float32))*tl.where(u>20.,u,tl.log(1.+tl.exp(tl.minimum(u,20.))))
        beta=tl.sigmoid(beta)
    if NORMALIZE:
        q*=tl.rsqrt(tl.sum(q*q,0)+1e-6)
        k*=tl.rsqrt(tl.sum(k*k,0)+1e-6)
    q*=SCALE
    ptr=Page+((tl.maximum(slot,0)*H+head)*K+i[:,None])*V+j[None,:]
    state=tl.load(ptr,slot>0,0).to(tl.float32)
    if KDA:retained=state*tl.exp(g)[:,None]
    else:retained=state*tl.exp(g)
    residual=v-tl.sum(k[:,None]*retained,0)
    x=retained+k[:,None]*(beta*residual[None,:])
    y=tl.sum(q[:,None]*x,0)
    tl.store(Out+(batch*H+head)*V+j,tl.where(slot>0,y,0.))
    tl.store(ptr,x,slot>0)
