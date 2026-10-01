"""Packed prefill codec, synchronous recurrent update, and input preparation.

Async decode uses readout.py -> rows.py -> fit.py without a dense state buffer.
This module retains the dense prefill encoder and the synchronous fallback.
"""
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice
from .bitpack import store_words
from .reductions import sum_pair


@tr.jit
def _half_scale(x):
    return tl.minimum(tl.maximum(x, 5.960464477539063e-8), 65504.).to(tl.float16).to(tl.float32)


@tr.jit
def _read(Page, Bits, Offsets, RowOffsets, ColOffsets, head, slot,
          PAGE, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, GROUP: tl.constexpr, KDA: tl.constexpr):
    i, j = tl.arange(0, K), tl.arange(0, V)
    b = tl.load(Bits + head*K + i)
    off = tl.load(Offsets + head*K + i)
    payload=off
    stride=tl.load(Offsets+H*K+(head*K+i)//K,(head*K+i)//K<H,0)
    roff = tl.load(RowOffsets + head*K + i)
    coff = tl.load(ColOffsets + head)
    base = Page + tl.maximum(slot, 0)*PAGE
    sb = tl.minimum(b, 8)
    bit = j[None, :]*sb[:, None]
    addr = payload[:,None]+(j[None,:]//min(4 if KDA else 16,V))*stride[:,None]+((j[None,:]%min(4 if KDA else 16,V))*sb[:,None])//8
    low = tl.load(base+addr, (b[:,None]!=16) & (slot>0), 0).to(tl.int32)
    high = tl.load(base+addr+1, (b[:,None]!=16) & (slot>0) & ((bit%8+sb[:,None])>8), 0).to(tl.int32)
    code = ((low | (high<<8)) >> (bit%8)) & ((1<<sb[:,None])-1)
    z = (code - ((1<<(sb[:,None]-1))-1)).to(tl.float32)
    rg = tl.full((K,V), 0, tl.int32)
    if KDA:
        z *= (1 << (8-sb[:,None])).to(tl.float32)
    else:
        z = tl.where(b[:,None]==2, 2.*code-3., z)
        rg = tl.where(b[:,None]==2, j[None,:]//GROUP, 0)
    r = tl.load((base.to(tl.pointer_type(tl.float16))+roff[:,None]//2+rg),
                (b[:,None]!=16) & (slot>0), 0).to(tl.float32)
    c = tl.load((base.to(tl.pointer_type(tl.float16))+coff//2+j), (coff>=0) & (slot>0), 0).to(tl.float32)
    pivot = tl.load((base.to(tl.pointer_type(tl.float16))+(payload[:,None]+(j[None,:]//min(4 if KDA else 16,V))*stride[:,None])//2+j[None,:]%min(4 if KDA else 16,V)),
                    (b[:,None]==16) & (slot>0), 0).to(tl.float32)
    return tl.where(b[:,None]==16, pivot, r*c[None,:]*z)


@tr.jit
def codec_kernel(Page, Bits, Offsets, RowOffsets, ColOffsets, Slots, State,
                 PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
                 GROUP:tl.constexpr,KDA:tl.constexpr):
    batch,head=tl.program_id(0),tl.program_id(1)
    slot=tl.load(Slots+batch)
    i,j=tl.arange(0,K),tl.arange(0,V)
    addr=State+((batch*H+head)*K+i[:,None])*V+j[None,:]
    x=_read(Page,Bits,Offsets,RowOffsets,ColOffsets,head,slot,PAGE,H,K,V,GROUP,KDA)
    tl.store(addr,x)


@tr.jit
def _column_kernel(State,Page,Bits,Impact,Offsets,RowOffsets,ColOffsets,Slots,
                  PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
                  GROUP:tl.constexpr,KDA:tl.constexpr,TILE:tl.constexpr,Partial=None,ROW_PARTIALS:tl.constexpr=False,ROWTILE:tl.constexpr=32,PRECISION:tl.constexpr=0):
    # Neighboring blocks touch adjacent columns of the same state page.
    batch,head,tile=tl.program_id(2),tl.program_id(1),tl.program_id(0)
    slot=tl.load(Slots+batch)
    i=tl.arange(0,K);j=tile*TILE+tl.arange(0,TILE)
    if tl.load(ColOffsets+head)>=0:
        b=tl.load(Bits+head*K+i) if KDA else tl.full((K,),tl.load(Bits+head*K) if PRECISION==0 else PRECISION,tl.int32)
        w=tl.load(Impact+head*K+i)
        roff=tl.load(RowOffsets+head*K+i);off=tl.load(Offsets+head*K+i)
        payload=off
        stride=tl.load(Offsets+H*K+(head*K+i)//K,(head*K+i)//K<H,0)
        coff=tl.load(ColOffsets+head)
        base=Page+tl.maximum(slot,0)*PAGE
        rg=tl.full((K,TILE),0,tl.int32)
        if not KDA:
            rg=tl.where(b[:,None]==2,j[None,:]//GROUP,0)
        if ROW_PARTIALS:
            groups=tl.arange(0,V//ROWTILE)
            sums=tl.load(Partial+((batch*H+head)*K+i[:,None])*(V//ROWTILE)+groups[None,:])
            r1=_half_scale(tl.sqrt(tl.maximum(tl.sum(sums,1)/V,1e-20)/w))
            r=tl.broadcast_to(r1[:,None],(K,TILE))
            if not KDA:
                group_sum=tl.full((K,TILE),0.,tl.float32)
                for t in tl.static_range(GROUP//ROWTILE):
                    group_sum+=tl.load(Partial+((batch*H+head)*K+i[:,None])*(V//ROWTILE)+(j[None,:]//GROUP)*(GROUP//ROWTILE)+t)
                r2=_half_scale(tl.sqrt(tl.maximum(group_sum/GROUP,1e-20)/w[:,None]))
                r=tl.where(b[:,None]==2,r2,r)
            row_mask=j[None,:]==0
            if not KDA:row_mask=tl.where(b[:,None]==2,j[None,:]%GROUP==0,row_mask)
            tl.store((base.to(tl.pointer_type(tl.float16))+roff[:,None]//2+rg),r,row_mask&(b[:,None]!=16)&(slot>0))
        else:
            r=tl.load((base.to(tl.pointer_type(tl.float16))+roff[:,None]//2+rg),(b[:,None]!=16)&(slot>0),1).to(tl.float32)
        x=tl.load(State+((batch*H+head)*K+i[:,None])*V+j[None,:]).to(tl.float32)
        qb=(1<<(tl.minimum(b,8)-1))-1
        norm=tl.full((K,),1,tl.int32)
        if KDA:
            norm=1<<(8-tl.minimum(b,8))
        else:
            qb=tl.where(b==2,3,qb)
        normalized=x/r/norm[:,None]
        c=_half_scale(tl.max(tl.where(b[:,None]!=16,tl.abs(normalized)*(1./qb.to(tl.float32))[:,None],0.),0))
        weights=tl.where(b!=16,w*w,0.)
        rn=r*norm[:,None]
        # Fit in normalized code units with the same weighted objective.
        weighted_r2=weights[:,None]*rn*rn
        weighted_x=weighted_r2*normalized
        inverse_column=1./c
        y=normalized*inverse_column[None,:]
        z=tl.minimum(tl.maximum(libdevice.nearbyint(y),-qb[:,None]),qb[:,None])
        if not KDA:
            z=tl.where(b[:,None]==2,tl.where(y>=0,1.,-1.)*tl.where(tl.abs(y)>=2,3.,1.),z)
        numerator,denominator=tl.reduce((weighted_x*z,weighted_r2*z*z),0,sum_pair)
        c=_half_scale(tl.where(denominator>0,numerator/tl.maximum(denominator,1e-30),c))
        inverse_column=1./c
        y=normalized*inverse_column[None,:]
        z=tl.minimum(tl.maximum(libdevice.nearbyint(y),-qb[:,None]),qb[:,None])
        if not KDA:
            z=tl.where(b[:,None]==2,tl.where(y>=0,1.,-1.)*tl.where(tl.abs(y)>=2,3.,1.),z)
        codes=z.to(tl.int32)+(1<<(tl.minimum(b[:,None],8)-1))-1
        if not KDA:
            codes=tl.where(b[:,None]==2,(z.to(tl.int32)+3)//2,codes)
        store_words(base,payload+(tile*TILE//min(4 if KDA else 16,V))*stride,tl.where(slot>0,b,0),codes,(tile*TILE)%min(4 if KDA else 16,V),K,TILE,Stride=stride if KDA else None)
        tl.store((base.to(tl.pointer_type(tl.float16))+coff//2+j),c,(coff>=0)&(slot>0))


@tr.jit
def column_kernel(State,Page,Bits,Impact,Offsets,RowOffsets,ColOffsets,Slots,
                  PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
                  GROUP:tl.constexpr,KDA:tl.constexpr,TILE:tl.constexpr,Partial=None,ROW_PARTIALS:tl.constexpr=False,ROWTILE:tl.constexpr=32,SPECIALIZE:tl.constexpr=True):
    if KDA or not SPECIALIZE:
        _column_kernel(State,Page,Bits,Impact,Offsets,RowOffsets,ColOffsets,Slots,PAGE,H,K,V,GROUP,KDA,TILE,Partial,ROW_PARTIALS,ROWTILE)
    else:
        precision=tl.load(Bits+tl.program_id(1)*K)
        if precision==2:
            _column_kernel(State,Page,Bits,Impact,Offsets,RowOffsets,ColOffsets,Slots,PAGE,H,K,V,GROUP,KDA,TILE,
                Partial,ROW_PARTIALS,ROWTILE,PRECISION=2)
        elif precision==4:
            _column_kernel(State,Page,Bits,Impact,Offsets,RowOffsets,ColOffsets,Slots,PAGE,H,K,V,GROUP,KDA,TILE,
                Partial,ROW_PARTIALS,ROWTILE,PRECISION=4)
        elif precision==6:
            _column_kernel(State,Page,Bits,Impact,Offsets,RowOffsets,ColOffsets,Slots,PAGE,H,K,V,GROUP,KDA,TILE,
                Partial,ROW_PARTIALS,ROWTILE,PRECISION=6)
        elif precision==8:
            _column_kernel(State,Page,Bits,Impact,Offsets,RowOffsets,ColOffsets,Slots,PAGE,H,K,V,GROUP,KDA,TILE,
                Partial,ROW_PARTIALS,ROWTILE,PRECISION=8)
        elif precision==16:
            _column_kernel(State,Page,Bits,Impact,Offsets,RowOffsets,ColOffsets,Slots,PAGE,H,K,V,GROUP,KDA,TILE,
                Partial,ROW_PARTIALS,ROWTILE,PRECISION=16)


@tr.jit
def prepare_inputs_kernel(Q,Key,G,Beta,ALog,Bias,PreparedQ,PreparedK,Decay,PreparedBeta,
                          H:tl.constexpr,HK:tl.constexpr,K:tl.constexpr,KDA:tl.constexpr,
                          QS:tl.constexpr,KS:tl.constexpr,NORMALIZE:tl.constexpr,
                          RAW_GATES:tl.constexpr,SCALE:tl.constexpr):
    head,batch=tl.program_id(0),tl.program_id(1)
    i=tl.arange(0,K)
    q=tl.load(Q+batch*QS+head//(H//HK)*K+i).to(tl.float32)
    k=tl.load(Key+batch*KS+head//(H//HK)*K+i).to(tl.float32)
    if NORMALIZE:
        q*=tl.rsqrt(tl.sum(q*q,0)+1e-6)
        k*=tl.rsqrt(tl.sum(k*k,0)+1e-6)
    gi=(batch*H+head)*K+i if KDA else batch*H+head
    g=tl.load(G+gi).to(tl.float32)
    beta=tl.load(Beta+batch*H+head).to(tl.float32)
    if RAW_GATES:
        bias=tl.load(Bias+head*K+i if KDA else Bias+head).to(tl.float32)
        u=g+bias
        g=-tl.exp(tl.load(ALog+head).to(tl.float32))*tl.where(u>20.,u,tl.log(1.+tl.exp(tl.minimum(u,20.))))
        beta=tl.sigmoid(beta)
    idx=(batch*H+head)*K+i
    tl.store(PreparedQ+idx,q*SCALE)
    tl.store(PreparedK+idx,k)
    tl.store(Decay+idx,tl.exp(g))
    tl.store(PreparedBeta+batch*H+head,beta)


@tr.jit
def update_kernel(Page,Bits,Offsets,RowOffsets,ColOffsets,Slots,Q,Key,Val,G,Beta,Out,State,ALog,Bias,
                  PAGE,H:tl.constexpr,HK:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
                  GROUP:tl.constexpr,KDA:tl.constexpr,NORMALIZE:tl.constexpr,RAW_GATES:tl.constexpr,
                  SCALE:tl.constexpr,QS:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr,TILE:tl.constexpr,Partial,WIDTH:tl.constexpr=0,HeadIds=None):
    # Neighboring blocks touch adjacent columns of the same state page.
    batch,head,tile=tl.program_id(2),tl.program_id(1),tl.program_id(0)
    if HeadIds is not None:
        head=tl.load(HeadIds+head)
    slot=tl.load(Slots+batch)
    i=tl.arange(0,K);j=tile*TILE+tl.arange(0,TILE)
    b=tl.load(Bits+head*K+i) if KDA else tl.full((K,),tl.load(Bits+head*K) if WIDTH==0 else WIDTH,tl.int32)
    off=tl.load(Offsets+head*K+i)
    payload=off
    stride=tl.load(Offsets+H*K+(head*K+i)//K,(head*K+i)//K<H,0)
    roff=tl.load(RowOffsets+head*K+i);coff=tl.load(ColOffsets+head)
    base=Page+tl.maximum(slot,0)*PAGE
    sb=tl.minimum(b,8);bit=j[None,:]*sb[:,None]
    addr=payload[:,None]+(j[None,:]//min(4 if KDA else 16,V))*stride[:,None]+((j[None,:]%min(4 if KDA else 16,V))*sb[:,None])//8
    low=tl.load(base+addr,(b[:,None]!=16)&(slot>0),0).to(tl.int32)
    high=tl.load(base+addr+1,(b[:,None]!=16)&(slot>0)&(bit%8+sb[:,None]>8),0).to(tl.int32)
    code=((low|(high<<8))>>(bit%8))&((1<<sb[:,None])-1)
    z=(code-((1<<(sb[:,None]-1))-1)).to(tl.float32)
    rg=tl.full((K,TILE),0,tl.int32)
    if KDA:
        z*=(1<<(8-sb[:,None])).to(tl.float32)
    else:
        z=tl.where(b[:,None]==2,2.*code-3.,z)
        rg=tl.where(b[:,None]==2,j[None,:]//GROUP,0)
    r=tl.load((base.to(tl.pointer_type(tl.float16))+roff[:,None]//2+rg),(b[:,None]!=16)&(slot>0),0).to(tl.float32)
    c=tl.load((base.to(tl.pointer_type(tl.float16))+coff//2+j),(coff>=0)&(slot>0),0).to(tl.float32)
    pivot=tl.load((base.to(tl.pointer_type(tl.float16))+(payload[:,None]+(j[None,:]//min(4 if KDA else 16,V))*stride[:,None])//2+j[None,:]%min(4 if KDA else 16,V)),(b[:,None]==16)&(slot>0),0).to(tl.float32)
    state=tl.where(b[:,None]==16,pivot,r*c[None,:]*z)
    q=tl.load(Q+batch*QS+head//(H//HK)*K+i).to(tl.float32)
    k=tl.load(Key+batch*KS+head//(H//HK)*K+i).to(tl.float32)
    val=tl.load(Val+batch*VS+head*V+j).to(tl.float32)
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
        u=g+bias
        g=-tl.exp(tl.load(ALog+head).to(tl.float32))*tl.where(u>20.,u,tl.log(1.+tl.exp(tl.minimum(u,20.))))
        beta=tl.sigmoid(beta)
    if NORMALIZE:
        q*=tl.rsqrt(tl.sum(q*q,0)+1e-6)
        k*=tl.rsqrt(tl.sum(k*k,0)+1e-6)
    q*=SCALE
    if KDA:
        retained=state*tl.exp(g)[:,None]
    else:
        retained=state*tl.exp(g)
    residual=val-tl.sum(k[:,None]*retained,0)
    x=retained+k[:,None]*(beta*residual[None,:])
    y=tl.sum(q[:,None]*x,0)
    tl.store(Out+(batch*H+head)*V+j,tl.where(slot>0,y,0.))
    # FP16 heads update their persistent pages directly; integer heads are fitted
    # from the temporary state by the synchronous/prefill codec.
    if WIDTH != 16:
        tl.store(State+((batch*H+head)*K+i[:,None])*V+j[None,:],x)
        tl.store(Partial+((batch*H+head)*K+i)*(V//TILE)+tile,tl.sum(tl.abs(x),1))
    tl.store(base.to(tl.pointer_type(tl.float16))+(payload[:,None]+(j[None,:]//min(4 if KDA else 16,V))*stride[:,None])//2+j[None,:]%min(4 if KDA else 16,V),x,(b[:,None]==16)&(slot>0))


@tr.jit
def rows_kernel(State,Page,Bits,Impact,Offsets,RowOffsets,Slots,
                PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,GROUP:tl.constexpr,
                KDA:tl.constexpr,ROWS:tl.constexpr=8,DSQ:tl.constexpr=False,TILED:tl.constexpr=False):
    row=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
    batch=tl.program_id(1)
    slot=tl.load(Slots+batch)
    j=tl.arange(0,V)
    b=tl.load(Bits+row,row<H*K,16)
    w=tl.load(Impact+row,row<H*K,1.)
    if DSQ:w=tl.full((ROWS,),1.,tl.float32)
    x=tl.load(State+(batch*H*K+row[:,None])*V+j[None,:],row[:,None]<H*K,0).to(tl.float32)
    base=Page+tl.maximum(slot,0)*PAGE
    off=tl.load(Offsets+row,row<H*K,0)
    roff=tl.load(RowOffsets+row,row<H*K,0)
    if TILED:
        payload=off
        stride=tl.load(Offsets+H*K+row//K,row//K<H,0)
        pivot=(payload[:,None]+(j[None,:]//min(4 if KDA else 16,V))*stride[:,None])//2+j[None,:]%min(4 if KDA else 16,V)
    else:
        pivot=off[:,None]//2+j[None,:]
    tl.store(base.to(tl.pointer_type(tl.float16))+pivot,x,
             (row[:,None]<H*K)&(b[:,None]==16)&(slot>0))
    r=_half_scale(tl.sqrt(tl.maximum(tl.sum(tl.abs(x),1)/V,1e-20)/w))
    tl.store((base.to(tl.pointer_type(tl.float16))+roff//2),r,
             (row<H*K)&(b!=16)&(slot>0)&((b!=2)|KDA))
    if not KDA:
        grouped=tl.reshape(tl.abs(x),(ROWS,V//GROUP,GROUP))
        r2=_half_scale(tl.sqrt(tl.maximum(tl.sum(grouped,2)/GROUP,1e-20)/w[:,None]))
        group=tl.arange(0,V//GROUP)
        tl.store((base.to(tl.pointer_type(tl.float16))+roff[:,None]//2+group[None,:]),r2,
                 (row[:,None]<H*K)&(b[:,None]==2)&(slot>0))
