"""One weighted column fit and code stores from affine row products."""
from .packed_read import load_four, load_words
from triton.experimental import gluon as tr
from triton.experimental.gluon import language as tl
from triton.experimental.gluon.language.extra import libdevice


@tr.jit
def pair_sum(a,b,c,d):
    return a+c,b+d


@tr.jit
def _fit(Page,Bits,Offsets,Columns,Slots,Coefficients,Delta,
        PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,GROUP:tl.constexpr,
        KDA:tl.constexpr,TILE:tl.constexpr=16,NW:tl.constexpr=4,HeadIds=None,WIDTH:tl.constexpr=0,FAST_RATIO:tl.constexpr=False,BYTE:tl.constexpr=False,State=None,Rows=None,Impact=None):
    KL:tl.constexpr=32 if KDA else 8
    VL:tl.constexpr=1 if KDA else 4
    WK:tl.constexpr=1 if KDA else NW
    WV:tl.constexpr=NW if KDA else 1
    # Keep byte loads vectorized while reducing the number of shuffled columns.
    KEY_LANES:tl.constexpr=(16 if KDA else 8) if State is None else 32
    layout:tl.constexpr=tl.BlockedLayout([4,1],[KEY_LANES,32//KEY_LANES],[1,NW],[0,1]) if BYTE else tl.BlockedLayout([1,4],[KL,VL],[WK,WV],[0,1])
    batch,head,tile=tl.program_id(2),tl.program_id(1),tl.program_id(0)
    if HeadIds is not None:
        head=tl.load(HeadIds+head)
    column=tl.load(Columns+head)
    if column>=0:
        i=tl.arange(0,K,layout=tl.SliceLayout(1,layout))
        j=tile*TILE+tl.arange(0,TILE,layout=tl.SliceLayout(0,layout))
        slot=tl.load(Slots+batch)
        base=Page+tl.maximum(slot,0)*PAGE
        half=base.to(tl.pointer_type(tl.float16))
        bits=tl.load(Bits+head*K+i) if KDA else tl.full((K,),WIDTH if WIDTH else tl.load(Bits+head*K),tl.int32,tl.SliceLayout(1,layout))
        width=tl.minimum(bits,8)
        off=tl.load(Offsets+head*K+i)
        stride=tl.load(Offsets+H*K+head)
        if BYTE:stride=tl.multiple_of(stride,4)
        limit=(1<<(width-1))-1
        if not KDA:limit=tl.where(bits==2,3,limit)
        group=tl.full((K,TILE),0,tl.int32,layout)
        if not KDA:
            group=tl.where(bits[:,None]==2,j[None,:]//GROUP,0)
        if State is None:
            if BYTE:
                z=tl.load(base.to(tl.pointer_type(tl.int8))+stride+i[:,None]+j[None,:]*K,
                          (slot>0),0).to(tl.float32)
            else:
                addr=off[:,None]+j[None,:]//min(4 if KDA else 16,V)*stride+(j[None,:]%min(4 if KDA else 16,V))*width[:,None]//8
                shift=j[None,:]*width[:,None]%8
                mask=(bits[:,None]!=16)&(slot>0)
                if KDA:
                    code=load_four(base,off,bits,stride,slot,tile,K,TILE,NW,layout)
                elif K>=4 and V>=16:
                    code=load_words(base,off,stride,slot,tile,WIDTH if WIDTH else tl.load(Bits+head*K),K,TILE,NW,layout,KEY_LANES=8)
                else:
                    lo=tl.load(base+addr,mask,0).to(tl.int32)
                    hi=tl.load(base+addr+1,mask&(shift+width[:,None]>8),0).to(tl.int32)
                    code=((lo|(hi<<8))>>shift)&((1<<width[:,None])-1)

                z=(code-limit[:,None]).to(tl.float32)
                if KDA:
                    z*=(1<<(8-width[:,None])).to(tl.float32)
                else:
                    z=tl.where(bits[:,None]==2,2.*code-3.,z)
            cs=H*K*(V//GROUP)
            idx=batch*3*cs+head*K*(V//GROUP)+group*K+i[:,None]
            key=tl.load(Coefficients+idx)
            retained=tl.load(Coefficients+cs+idx)
            metric=tl.load(Coefficients+2*cs+idx)
            old=tl.load(half+column//2+j,slot>0,0).to(tl.float32)
            delta=tl.load(Delta+(batch*H+head)*V+j)
            x=(z*old[None,:])*retained+key*delta[None,:]
            x=tl.where(bits[:,None]!=16,x,0.)
        else:
            ro=tl.load(Rows+head*K+i)
            r=tl.load(half+ro[:,None]//2+group,(bits[:,None]!=16)&(slot>0),1.).to(tl.float32)
            norm=(1<<(8-width)).to(tl.float32) if KDA else tl.full((K,),1.,tl.float32,tl.SliceLayout(1,layout))
            rn=r*norm[:,None]
            w=tl.load(Impact+head*K+i)
            metric=tl.where(bits[:,None]!=16,w[:,None]*w[:,None]*rn*rn,0.)
            x=tl.load(State+((batch*H+head)*K+i[:,None])*V+j[None,:]).to(tl.float32)*libdevice.fast_dividef(1.,rn)
            x=tl.where(bits[:,None]!=16,x,0.)
        if not KDA and WIDTH:
            bound:tl.constexpr=3 if WIDTH==2 else (1<<(WIDTH-1))-1
            c=tl.max(tl.abs(x),0)*(1./bound)
        else:
            c=tl.max(tl.abs(x)*libdevice.fast_dividef(1.,limit.to(tl.float32))[:,None],0)
        c=tl.minimum(tl.maximum(c,2.**-24),65504.).to(tl.float16).to(tl.float32)
        # Stored positive FP16 scales fit the fast reciprocal range; it needs
        # no general float32 exponent-range handling here.
        y=x*libdevice.fast_dividef(1.,c)[None,:]
        if BYTE and KDA:
            z=libdevice.float2int_rn(tl.minimum(tl.maximum(y,-limit[:,None]),limit[:,None])).to(tl.float32)
        else:
            z=tl.minimum(tl.maximum(libdevice.nearbyint(y),-limit[:,None]),limit[:,None])
        if not KDA:
            z=tl.where(bits[:,None]==2,tl.where(y>=0,1.,-1.)*tl.where(tl.abs(y)>=2,3.,1.),z)
        if FAST_RATIO:
            weighted_code=metric*z
            num,den=tl.reduce((weighted_code*x,weighted_code*z),0,pair_sum)
        else:
            num,den=tl.reduce((metric*x*z,metric*z*z),0,pair_sum)
        ratio=libdevice.fast_dividef(num,tl.maximum(den,1e-30)) if FAST_RATIO else num/tl.maximum(den,1e-30)
        c=tl.where(den>0,ratio,c)
        c=tl.minimum(tl.maximum(c,2.**-24),65504.).to(tl.float16).to(tl.float32)
        y=x*libdevice.fast_dividef(1.,c)[None,:]
        z=libdevice.float2int_rn(tl.minimum(tl.maximum(y,-limit[:,None]),limit[:,None]))
        if not KDA:
            z=tl.where(bits[:,None]==2,tl.where(y>=0,1,-1)*tl.where(tl.abs(y)>=2,3,1),z)
        if BYTE:
            if KDA:
                z=z<<(8-width[:,None])
            tl.store(base.to(tl.pointer_type(tl.int8))+stride+i[:,None]+j[None,:]*K,
                     z.to(tl.int8),slot>0)
        else:
            codes=z.to(tl.int32)+(1<<(width[:,None]-1))-1
            if not KDA:
                codes=tl.where(bits[:,None]==2,(z.to(tl.int32)+3)//2,codes)
            # Four adjacent values share a lane, so packing needs no warp shuffle.
            grouped=tl.reshape(codes,(K,TILE//4,4))
            grouped=tl.convert_layout(grouped,tl.BlockedLayout([1,1,4],[KL,VL,1],[WK,WV,1],[0,2,1]))
            gl:tl.constexpr=tl.SliceLayout(0,tl.SliceLayout(0,tl.BlockedLayout([1,1,4],[KL,VL,1],[WK,WV,1],[0,2,1])))
            lane=tl.arange(0,4,layout=gl)
            wl:tl.constexpr=tl.SliceLayout(1,tl.SliceLayout(2,tl.BlockedLayout([1,1,4],[KL,VL,1],[WK,WV,1],[0,2,1])))
            w=tl.convert_layout(width,wl)[:,None,None]
            word=tl.sum(grouped<<(tl.expand_dims(tl.expand_dims(lane,0),0)*w),2)
            pl:tl.constexpr=tl.BlockedLayout([1,1],[KL,VL],[WK,WV],[0,1])
            word=tl.convert_layout(word,pl)
            off=tl.convert_layout(off,tl.SliceLayout(1,pl))
            bits=tl.convert_layout(bits,tl.SliceLayout(1,pl))
            ol:tl.constexpr=tl.SliceLayout(0,tl.BlockedLayout([1,1],[KL,VL],[WK,WV],[0,1]))
            g=tl.arange(0,TILE//4,layout=ol)
            rowoff=off[:,None]+((tile*TILE+g[None,:]*4)//min(4 if KDA else 16,V))*stride
            start=((tile*TILE+g[None,:]*4)%min(4 if KDA else 16,V))//4
            if KDA:
                # Mixed row widths can have odd byte offsets; byte stores preserve
                # tight packing without padding integer rows to half-word alignment.
                for byte in tl.static_range(4):
                    tl.store(base+rowoff+byte,(word>>(8*byte)).to(tl.uint8),
                             (bits[:,None]!=16)&(slot>0)&(byte<bits[:,None]//2))
            else:
                tl.store(base+rowoff+start,word.to(tl.uint8),(bits[:,None]==2)&(slot>0))
                tl.store(base.to(tl.pointer_type(tl.uint16))+rowoff//2+start,word.to(tl.uint16),(bits[:,None]==4)&(slot>0))
                tl.store(base.to(tl.pointer_type(tl.uint16))+rowoff//2+start*2,word.to(tl.uint16),(bits[:,None]==8)&(slot>0))
                tl.store(base.to(tl.pointer_type(tl.uint16))+rowoff//2+start*2+1,(word>>16).to(tl.uint16),(bits[:,None]==8)&(slot>0))
                for byte in tl.static_range(3):
                    tl.store(base+rowoff+start*3+byte,(word>>(8*byte)).to(tl.uint8),(bits[:,None]==6)&(slot>0))
        tl.store(half+column//2+j,c,slot>0)


@tr.jit
def fit(Page,Bits,Offsets,Columns,Slots,Coefficients,Delta,
        PAGE,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,GROUP:tl.constexpr,
        KDA:tl.constexpr,TILE:tl.constexpr=16,NW:tl.constexpr=4,HeadIds=None,WIDTH:tl.constexpr=0,FAST_RATIO:tl.constexpr=False,BYTE:tl.constexpr=False,State=None,Rows=None,Impact=None,SPECIALIZE:tl.constexpr=False):
    if SPECIALIZE:
        width=tl.load(Bits+tl.program_id(1)*K)
        if width==2:
            _fit(Page,Bits,Offsets,Columns,Slots,Coefficients,Delta,PAGE,H,K,V,GROUP,KDA,TILE,NW,WIDTH=2,FAST_RATIO=FAST_RATIO,BYTE=BYTE,State=State,Rows=Rows,Impact=Impact)
        elif width==4:
            _fit(Page,Bits,Offsets,Columns,Slots,Coefficients,Delta,PAGE,H,K,V,GROUP,KDA,TILE,NW,WIDTH=4,FAST_RATIO=FAST_RATIO,BYTE=BYTE,State=State,Rows=Rows,Impact=Impact)
        elif width==6:
            _fit(Page,Bits,Offsets,Columns,Slots,Coefficients,Delta,PAGE,H,K,V,GROUP,KDA,TILE,NW,WIDTH=6,FAST_RATIO=FAST_RATIO,BYTE=BYTE,State=State,Rows=Rows,Impact=Impact)
        elif width==8:
            _fit(Page,Bits,Offsets,Columns,Slots,Coefficients,Delta,PAGE,H,K,V,GROUP,KDA,TILE,NW,WIDTH=8,FAST_RATIO=FAST_RATIO,BYTE=BYTE,State=State,Rows=Rows,Impact=Impact)
    else:
        _fit(Page,Bits,Offsets,Columns,Slots,Coefficients,Delta,PAGE,H,K,V,GROUP,KDA,TILE,NW,HeadIds=HeadIds,WIDTH=WIDTH,FAST_RATIO=FAST_RATIO,BYTE=BYTE,State=State,Rows=Rows,Impact=Impact)
