"""GDN and KDA readers retaining row metadata across value segments."""
from .packed_read import load_words, load_four
from triton.experimental import gluon as tr
from triton.experimental.gluon import language as tl

@tr.jit
def readout(Page,Bits,Offsets,Rows,Columns,Slots,Q,Key,Val,Decay,Beta,Out,Delta,Partial,
            PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,GROUP:tl.constexpr,
            KDA:tl.constexpr,TILE:tl.constexpr,VS:tl.constexpr,
            HeadIds=None,WIDTH:tl.constexpr=0,NW:tl.constexpr=1,INTEGER_HEADS:tl.constexpr=False,BYTE:tl.constexpr=True,StableSlots=None,RawKey=None,RawDecay=None,ALog=None,Bias=None,HK:tl.constexpr=0,QS:tl.constexpr=0,KS:tl.constexpr=0,NORMALIZE:tl.constexpr=False,SCALE:tl.constexpr=1.,KEY_LANES:tl.constexpr=32,PART:tl.constexpr=8):
    tl.static_assert(not KDA and INTEGER_HEADS and WIDTH==0)
    tl.static_assert(TILE<=GROUP)
    L:tl.constexpr=tl.BlockedLayout([4,1],[32,1],[1,NW],[0,1]) if BYTE else tl.BlockedLayout([1,4],[KEY_LANES,32//KEY_LANES],[1,NW],[0,1])
    batch,head,tile=tl.program_id(2),tl.program_id(1),tl.program_id(0)
    if HeadIds is not None:head=tl.load(HeadIds+head)
    i=tl.arange(0,K,layout=tl.SliceLayout(1,L));idx=(batch*H+head)*K+i
    if RawKey is not None:
        q=tl.load(Q+batch*QS+head//(H//HK)*K+i).to(tl.float32)
        k=tl.load(Key+batch*KS+head//(H//HK)*K+i).to(tl.float32)
        if NORMALIZE:
            q*=tl.rsqrt(tl.sum(q*q,0)+1e-6);k*=tl.rsqrt(tl.sum(k*k,0)+1e-6)
        q*=SCALE
        g=tl.load(Decay+idx if KDA else Decay+batch*H+head).to(tl.float32)
        beta=tl.load(Beta+batch*H+head).to(tl.float32)
        if ALog is not None:
            bias=tl.load(Bias+head*K+i if KDA else Bias+head).to(tl.float32);u=g+bias
            g=-tl.exp(tl.load(ALog+head).to(tl.float32))*tl.where(u>20.,u,tl.log(1.+tl.exp(tl.minimum(u,20.))))
            beta=1./(1.+tl.exp(-beta))
        decay=tl.full((K,),1.,tl.float32,tl.SliceLayout(1,L))*tl.exp(g)
        if tile==0:
            tl.store(RawKey+idx,k);tl.store(RawDecay+idx,decay)
    else:
        q=tl.load(Q+idx);k=tl.load(Key+idx);decay=tl.load(Decay+idx);beta=tl.load(Beta+batch*H+head)
    slot=tl.load(Slots+batch)
    if StableSlots is not None:
        if (head==0)&(tile==0):tl.store(StableSlots+batch,slot)
    base=Page+tl.maximum(slot,0)*PAGE;half=base.to(tl.pointer_type(tl.float16))
    bits=tl.load(Bits+head*K)
    if BYTE:
        code_base=tl.multiple_of(tl.load(Offsets+H*K+head),4)
    else:
        offset=tl.multiple_of(tl.load(Offsets+head*K),2)+i*(min(16,V)*bits//8)
        stride=K*(min(16,V)*bits//8)
    column=tl.load(Columns+head);row=tl.load(Rows+head*K+i)
    rg=tl.where(bits==2,tile*TILE//GROUP,0)
    r=tl.load(half+row//2+rg,slot>0,0).to(tl.float32)*decay
    total=tl.full((K,),0.,tl.float32,tl.SliceLayout(1,L))
    PART_SIZE:tl.constexpr=min(PART,TILE)
    for segment in range(TILE//PART_SIZE):
        j=tile*TILE+segment*PART_SIZE+tl.arange(0,PART_SIZE,layout=tl.SliceLayout(0,L))
        if BYTE:
            z=tl.load(base.to(tl.pointer_type(tl.int8))+code_base+i[:,None]+j[None,:]*K,slot>0,0).to(tl.float32)
        else:
            code=load_words(base,offset,stride,slot,tile*(TILE//PART_SIZE)+segment,bits,K,PART_SIZE,NW,L,KEY_LANES=KEY_LANES)
            z=tl.where(bits==2,2*code-3,code-((1<<(bits-1))-1)).to(tl.float32)
        c=tl.load(half+column//2+j,slot>0,0).to(tl.float32)
        x=r[:,None]*c[None,:]*z
        value=tl.load(Val+batch*VS+head*V+j).to(tl.float32)
        delta=beta*(value-tl.sum(k[:,None]*x,0));x+=k[:,None]*delta[None,:]
        total+=tl.sum(tl.abs(x),1)
        output=tl.sum(q[:,None]*x,0)
        tl.store(Out+(batch*H+head)*V+j,tl.where(slot>0,output,0.))
        tl.store(Delta+(batch*H+head)*V+j,delta)
    tl.store(Partial+((batch*H+head)*(V//TILE)+tile)*K+i,total)


@tr.jit
def readout_kda(Page,Bits,Offsets,Rows,Columns,Slots,Q,Key,Val,Decay,Beta,Out,Delta,Partial,StableSlots,
                PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,TILE:tl.constexpr,VS:tl.constexpr):
    # Reuse row metadata across value segments and emit one row sum per tile.
    NW:tl.constexpr=1
    PART:tl.constexpr=4
    L:tl.constexpr=tl.BlockedLayout([1,4],[32,1],[1,NW],[0,1])
    tile,head,batch=tl.program_id(2),tl.program_id(1),tl.program_id(0)
    i=tl.arange(0,K,layout=tl.SliceLayout(1,L));idx=(batch*H+head)*K+i
    slot=tl.load(Slots+batch)
    if StableSlots is not None:
        if head==0 and tile==0:tl.store(StableSlots+batch,slot)
    q=tl.load(Q+idx);k=tl.load(Key+idx);decay=tl.load(Decay+idx);beta=tl.load(Beta+batch*H+head)
    bits=tl.load(Bits+head*K+i);width=tl.minimum(bits,8)
    off=tl.load(Offsets+head*K+i).to(tl.uint32);stride=tl.load(Offsets+H*K+head).to(tl.uint32)
    row=tl.load(Rows+head*K+i);column=tl.load(Columns+head)
    base=Page+tl.maximum(slot,0)*PAGE;half=base.to(tl.pointer_type(tl.float16))
    r=tl.load(half+(row>>1),(slot>0)&(bits!=16),0).to(tl.float32)*(1<<(8-width)).to(tl.float32)*decay
    limit=(1<<(width-1))-1
    total=tl.full((K,),0.,tl.float32,tl.SliceLayout(1,L))
    for segment in range(TILE//PART):
        start=tile*TILE+segment*PART
        j=start+tl.arange(0,PART,layout=tl.SliceLayout(0,L))
        code=load_four(base,off,bits,stride,slot,start//PART,K,PART,NW,L)
        z=(code-limit[:,None]).to(tl.float32)
        c=tl.load(half+(column>>1)+j,(slot>0)&(column>=0),0).to(tl.float32)
        x=r[:,None]*c[None,:]*z
        pivot_ptr=(off[:,None]+(j[None,:].to(tl.uint32)>>2)*stride)//2+(j[None,:]&3)
        pivot=tl.load(half+pivot_ptr,(slot>0)&(bits[:,None]==16),0).to(tl.float32)
        x=tl.where(bits[:,None]==16,pivot*decay[:,None],x)
        value=tl.load(Val+batch*VS+head*V+j).to(tl.float32)
        delta=beta*(value-tl.sum(k[:,None]*x,0))
        x+=k[:,None]*delta[None,:]
        tl.store(Out+(batch*H+head)*V+j,tl.where(slot>0,tl.sum(q[:,None]*x,0),0.))
        tl.store(Delta+(batch*H+head)*V+j,delta)
        total+=tl.sum(tl.abs(x),1)
        tl.store(half+pivot_ptr,x,(slot>0)&(bits[:,None]==16))
    tl.store(Partial+((batch*H+head)*(V//TILE)+tile)*K+i,total)
