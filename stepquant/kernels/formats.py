"""Fused decode/update/readout/writeback for state formats.

A program owns one request/head: no dense persistent shadow, no packing races.
The existing tiled STEPQuant kernel remains the default paper implementation.
"""
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .fused import _half_scale
from .bitpack import store_words


@tr.jit
def read(Page, Bits, Offsets, Rows, Columns, head, slot,
         PAGE, K: tl.constexpr, V: tl.constexpr, GROUP: tl.constexpr,
         KIND: tl.constexpr, START=0, WIDTH: tl.constexpr=0, FULL_V:tl.constexpr=0):
    i,j = tl.arange(0,K),START+tl.arange(0,V)
    b = tl.full((K,),WIDTH,tl.int32) if WIDTH>0 else tl.load(Bits+head*K+i)
    off = (head*K+i)*(FULL_V*WIDTH//8) if WIDTH>0 else tl.load(Offsets+head*K+i)
    base = Page+tl.maximum(slot,0)*PAGE
    if KIND == 'fp32':
        x=tl.load((base+off[:,None]+4*j[None,:]).to(tl.pointer_type(tl.float32)),slot>0,0)
    else:
        sb=tl.minimum(b,8)
        bit=j[None,:]*sb[:,None]
        addr=off[:,None]+bit//8
        lo=tl.load(base+addr,(b[:,None]!=16)&(slot>0),0).to(tl.int32)
        hi=tl.load(base+addr+1,(b[:,None]!=16)&(slot>0)&(bit%8+sb[:,None]>8),0).to(tl.int32)
        code=((lo|(hi<<8))>>(bit%8))&((1<<sb[:,None])-1)
        ro=tl.load(Rows+head*K+i)
        if KIND == 'dsq':
            r=tl.load((base.to(tl.pointer_type(tl.float16))+ro[:,None]//2),(slot>0)&(b[:,None]!=16),0).to(tl.float32)
            co=tl.load(Columns+head)
            c=tl.load((base.to(tl.pointer_type(tl.float16))+co//2+2*j),slot>0,0).to(tl.float32)
            x=r*c[None,:]*(code-((1<<(sb[:,None]-1))-1)).to(tl.float32)
        else:
            meta=base+ro[:,None]+4*(j[None,:]//GROUP)
            s=tl.load(base.to(tl.pointer_type(tl.float16))+ro[:,None]//2+2*(j[None,:]//GROUP),(slot>0)&(b[:,None]!=16),0).to(tl.float32)
            z=tl.load(base.to(tl.pointer_type(tl.float16))+ro[:,None]//2+2*(j[None,:]//GROUP)+1,(slot>0)&(b[:,None]!=16),0).to(tl.float32)
            x=(code-z)*s
        pivot=tl.load((base.to(tl.pointer_type(tl.float16))+off[:,None]//2+j[None,:]),(b[:,None]==16)&(slot>0),0).to(tl.float32)
        x=tl.where(b[:,None]==16,pivot,x)
    return x


@tr.jit
def store(x,Page,Bits,Impact,Offsets,Rows,Columns,head,slot,
          PAGE,K:tl.constexpr,V:tl.constexpr,GROUP:tl.constexpr,
          KIND:tl.constexpr,START=0, WIDTH:tl.constexpr=0, ROW_READY:tl.constexpr=False, FULL_V:tl.constexpr=0):
    i,j=tl.arange(0,K),START+tl.arange(0,V)
    b=tl.full((K,),WIDTH,tl.int32) if WIDTH>0 else tl.load(Bits+head*K+i)
    off=(head*K+i)*(FULL_V*WIDTH//8) if WIDTH>0 else tl.load(Offsets+head*K+i)
    base=Page+tl.maximum(slot,0)*PAGE
    if KIND == 'fp32':
        tl.store((base+off[:,None]+4*j[None,:]).to(tl.pointer_type(tl.float32)),x,slot>0)
    else:
        sb=tl.minimum(b,8)
        integer=b!=16
        ro=tl.load(Rows+head*K+i)
        if KIND == 'dsq':
            w=tl.full((K,),1.,tl.float32)
            if ROW_READY:
                r=tl.load((base.to(tl.pointer_type(tl.float16))+ro//2),(slot>0)&integer,1.).to(tl.float32)
            else:
                r=_half_scale(tl.sqrt(tl.maximum(tl.sum(tl.abs(x),1)/V,1e-20)/w))
            y=x/r[:,None]
            levels=(1<<(sb-1))-1
            c=_half_scale(tl.max(tl.where(integer[:,None],tl.abs(y)/levels[:,None],0.),0))
            z=tl.full((V,),0.,tl.float32)
            codes=tl.minimum(tl.maximum(libdevice.nearbyint(y/c[None,:]),-levels[:,None]),levels[:,None])+levels[:,None]
            co=tl.load(Columns+head)
            if not ROW_READY:
                tl.store((base.to(tl.pointer_type(tl.float16))+ro//2),r,(slot>0)&integer)
            tl.store((base.to(tl.pointer_type(tl.float16))+co//2+2*j),c,slot>0)
            tl.store((base.to(tl.pointer_type(tl.float16))+co//2+2*j+1),z,slot>0)
        else:
            y=x
            grouped=tl.reshape(y,(K,V//GROUP,GROUP))
            limit=(1<<(sb-1))-1
            s=_half_scale(tl.max(tl.abs(grouped),2)/limit[:,None])
            z=tl.broadcast_to(limit[:,None],(K,V//GROUP)).to(tl.float32)
            expanded_s=tl.reshape(tl.broadcast_to(s[:,:,None],(K,V//GROUP,GROUP)),(K,V))
            expanded_z=tl.reshape(tl.broadcast_to(z[:,:,None],(K,V//GROUP,GROUP)),(K,V))
            limit=(1<<(sb-1))-1
            codes=tl.minimum(tl.maximum(libdevice.nearbyint(tl.div_rn(y,expanded_s)),-limit[:,None]),limit[:,None])+expanded_z
            groups=START//GROUP+tl.arange(0,V//GROUP)
            meta=base+ro[:,None]+4*groups[None,:]
            tl.store(base.to(tl.pointer_type(tl.float16))+ro[:,None]//2+2*groups[None,:],s,(slot>0)&integer[:,None])
            tl.store(base.to(tl.pointer_type(tl.float16))+ro[:,None]//2+2*groups[None,:]+1,z,(slot>0)&integer[:,None])
        codes=codes.to(tl.int32)
        store_words(base,off,tl.where(slot>0,b,0),codes,START,K,V)
        tl.store((base.to(tl.pointer_type(tl.float16))+off[:,None]//2+j[None,:]),x,(b[:,None]==16)&(slot>0))


@tr.jit
def codec(Page,Bits,Impact,Offsets,Rows,Columns,Slots,State,
          PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,GROUP:tl.constexpr,
          KIND:tl.constexpr,ENCODE:tl.constexpr,TILE:tl.constexpr=0,WIDTH:tl.constexpr=0,ROW_READY:tl.constexpr=False):
    batch,head=tl.program_id(0),tl.program_id(1)
    slot=tl.load(Slots+batch)
    BLOCK:tl.constexpr=V if TILE==0 else TILE
    start=tl.program_id(2)*BLOCK
    i,j=tl.arange(0,K),start+tl.arange(0,BLOCK)
    addr=State+((batch*H+head)*K+i[:,None])*V+j[None,:]
    if ENCODE:
        x=tl.load(addr).to(tl.float32)
        store(x,Page,Bits,Impact,Offsets,Rows,Columns,head,slot,PAGE,K,BLOCK,GROUP,KIND,start,WIDTH,ROW_READY,V)
    else:
        x=read(Page,Bits,Offsets,Rows,Columns,head,slot,PAGE,K,BLOCK,GROUP,KIND,start,WIDTH,V)
        tl.store(addr,x)


@tr.jit
def decode(Page,Bits,Impact,Offsets,Rows,Columns,Slots,Q,Key,Val,G,Beta,Out,ALog,Bias,
           PAGE,H:tl.constexpr,HK:tl.constexpr,K:tl.constexpr,V:tl.constexpr,GROUP:tl.constexpr,
           KIND:tl.constexpr,KDA:tl.constexpr,NORMALIZE:tl.constexpr,
           RAW_GATES:tl.constexpr,SCALE:tl.constexpr,QS:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr,TILE:tl.constexpr=0,WIDTH:tl.constexpr=0,State=None,UPDATE_ONLY:tl.constexpr=False):
    batch,head=tl.program_id(0),tl.program_id(1)
    slot=tl.load(Slots+batch)
    BLOCK:tl.constexpr=V if TILE==0 else TILE
    start=tl.program_id(2)*BLOCK
    i,j=tl.arange(0,K),start+tl.arange(0,BLOCK)
    q=tl.load(Q+batch*QS+head//(H//HK)*K+i).to(tl.float32)
    k=tl.load(Key+batch*KS+head//(H//HK)*K+i).to(tl.float32)
    v=tl.load(Val+batch*VS+head*V+j).to(tl.float32)
    beta=tl.load(Beta+batch*H+head).to(tl.float32)
    if KDA:
        g=tl.load(G+(batch*H+head)*K+i).to(tl.float32)
    else:
        g=tl.load(G+batch*H+head).to(tl.float32)
    if RAW_GATES:
        if KDA:
            bias=tl.load(Bias+head*K+i).to(tl.float32)
        else:
            bias=tl.load(Bias+head).to(tl.float32)
        a=tl.load(ALog+head).to(tl.float32)
        u=g+bias
        g=-tl.exp(a)*tl.where(u>20.,u,tl.log(1.+tl.exp(tl.minimum(u,20.))))
        beta=tl.sigmoid(beta)
    if NORMALIZE:
        q*=tl.rsqrt(tl.sum(q*q,0)+1e-6)
        k*=tl.rsqrt(tl.sum(k*k,0)+1e-6)
    q*=SCALE
    state=read(Page,Bits,Offsets,Rows,Columns,head,slot,PAGE,K,BLOCK,GROUP,KIND,start,WIDTH,V)
    if KDA:
        retained=state*tl.exp(g)[:,None]
    else:
        retained=state*tl.exp(g)
    residual=v-tl.sum(k[:,None]*retained,0)
    x=retained+k[:,None]*(beta*residual[None,:])
    y=tl.sum(q[:,None]*x,0)
    tl.store(Out+(batch*H+head)*V+j,tl.where(slot>0,y,0.))
    if UPDATE_ONLY:
        tl.static_assert(BLOCK==V)
        tl.store(State+((batch*H+head)*K+i[:,None])*V+j[None,:],x)
        b=tl.load(Bits+head*K+i)
        ro=tl.load(Rows+head*K+i)
        off=tl.load(Offsets+head*K+i)
        base=Page+tl.maximum(slot,0)*PAGE
        w=tl.load(Impact+head*K+i)
        if KIND=='dsq':w=tl.full((K,),1.,tl.float32)
        r=_half_scale(tl.sqrt(tl.maximum(tl.sum(tl.abs(x),1)/V,1e-20)/w))
        tl.store((base.to(tl.pointer_type(tl.float16))+ro//2),r,(slot>0)&(b!=16))
        tl.store((base.to(tl.pointer_type(tl.float16))+off[:,None]//2+j[None,:]),x,(slot>0)&(b[:,None]==16))
    else:
        store(x,Page,Bits,Impact,Offsets,Rows,Columns,head,slot,PAGE,K,BLOCK,GROUP,KIND,start,WIDTH,False,V)


@tr.jit
def codec_rows(Page,Bits,Impact,Offsets,Rows,Columns,Slots,State,
               PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,GROUP:tl.constexpr,
               KIND:tl.constexpr,ENCODE:tl.constexpr,
               TILE:tl.constexpr,WIDTH:tl.constexpr,ROW_BLOCK:tl.constexpr):
    # Groupwise codecs are independent across rows, unlike delta-rule readouts.
    block,batch,tile=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    slot=tl.load(Slots+batch)
    i=block*ROW_BLOCK+tl.arange(0,ROW_BLOCK)
    start=tile*TILE
    j=start+tl.arange(0,TILE)
    ptr=State+(batch*H*K+i[:,None])*V+j[None,:]
    if ENCODE:
        x=tl.load(ptr).to(tl.float32)
        store(x,Page,Bits,Impact,Offsets,Rows,Columns,block,slot,
              PAGE,ROW_BLOCK,TILE,GROUP,KIND,start,WIDTH,False,V)
    else:
        x=read(Page,Bits,Offsets,Rows,Columns,block,slot,
               PAGE,ROW_BLOCK,TILE,GROUP,KIND,start,WIDTH,V)
        tl.store(ptr,x)
