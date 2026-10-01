"""Register-local four-code loads for tight mixed-width KDA and uniform GDN pages."""
from triton.experimental import gluon as tr
from triton.experimental.gluon import language as tl


@tr.jit
def load_four(base,offset,bits,stride,slot,tile,K:tl.constexpr,TILE:tl.constexpr,
              NW:tl.constexpr,LAYOUT:tl.constexpr):
    packed:tl.constexpr=tl.BlockedLayout([1,1],[32,1],[1,NW],[0,1])
    expanded:tl.constexpr=tl.BlockedLayout([1,1,4],[32,1,1],[1,NW,1],[0,2,1])
    off=tl.convert_layout(offset,tl.SliceLayout(1,packed)).to(tl.uint32)
    b=tl.convert_layout(bits,tl.SliceLayout(1,packed))
    group=tile*(TILE//4)+tl.arange(0,TILE//4,layout=tl.SliceLayout(0,packed))
    ptr=base+off[:,None]+group[None,:]*stride
    mask=(b[:,None]!=16)&(slot>0)
    # Read each tight four-code word through aligned transactions, including
    # mixed-width rows whose first byte is unaligned. Metadata after the code
    # region keeps the final aligned load within its allocated page.
    address=ptr.to(tl.uint64)
    shift=((address&3)*8).to(tl.uint32)
    aligned=(address&~tl.full((),3,tl.uint64,tl.BlockedLayout([1,1],[32,1],[1,NW],[0,1]))).to(tl.pointer_type(tl.uint32))
    lo=tl.load(aligned,mask,0)
    hi=tl.load(aligned+1,mask&(shift+4*tl.minimum(b[:,None],8)>32),0)
    word=tl.inline_asm_elementwise("shf.r.wrap.b32 $0, $1, $2, $3;",constraints="=r,r,r,r",args=(lo,hi,shift),dtype=tl.uint32,is_pure=True,pack=1).to(tl.int32)
    word=tl.convert_layout(word,tl.SliceLayout(2,expanded))[:,:,None]
    width=tl.convert_layout(tl.minimum(b,8),tl.SliceLayout(1,tl.SliceLayout(2,expanded)))[:,None,None]
    lane=tl.arange(0,4,layout=tl.SliceLayout(0,tl.SliceLayout(0,expanded)))
    shift=tl.expand_dims(tl.expand_dims(lane,0),0)*width
    codes=(word>>shift)&((1<<width)-1)
    return tl.convert_layout(tl.reshape(codes,(K,TILE)),LAYOUT)


@tr.jit
def load_words(base,offset,stride,slot,tile,precision,K:tl.constexpr,TILE:tl.constexpr,NW:tl.constexpr,LAYOUT:tl.constexpr,KEY_LANES:tl.constexpr=4,KEY_WARPS:tl.constexpr=0):
    # One lane owns four codes; tight INT6 uses two aligned half-word loads.
    KW:tl.constexpr=KEY_WARPS if KEY_WARPS else NW
    P:tl.constexpr=tl.BlockedLayout([1,1],[KEY_LANES,32//KEY_LANES],[KW,NW//KW],[1,0])
    E:tl.constexpr=tl.BlockedLayout([1,1,4],[KEY_LANES,32//KEY_LANES,1],[KW,NW//KW,1],[1,0,2])
    off=tl.convert_layout(offset,tl.SliceLayout(1,P))
    group=tile*(TILE//4)+tl.arange(0,TILE//4,layout=tl.SliceLayout(0,P))
    addr=off[:,None]+(group[None,:]//4)*stride+(group[None,:]%4)*(precision//2)
    if precision==4:
        word=tl.load(base.to(tl.pointer_type(tl.uint16))+addr//2,slot>0,0).to(tl.int32)
    elif precision==8:
        word=tl.load(base.to(tl.pointer_type(tl.uint32))+addr//4,slot>0,0).to(tl.int32)
    else:
        if precision==6:
            pointer=(base+addr).to(tl.uint64)
            shift=((pointer&1)*8).to(tl.uint32)
            aligned=(pointer&~tl.full((),1,tl.uint64,P)).to(tl.pointer_type(tl.uint16))
            lo=tl.load(aligned,slot>0,0).to(tl.uint32)
            hi=tl.load(aligned+1,slot>0,0).to(tl.uint32)
            word=((lo|(hi<<16))>>shift).to(tl.int32)
        else:
            word=tl.load(base+addr,slot>0,0).to(tl.int32)
    word=tl.convert_layout(word,tl.SliceLayout(2,E))[:,:,None]
    lane=tl.arange(0,4,layout=tl.SliceLayout(0,tl.SliceLayout(0,E)))
    code=(word>>(tl.expand_dims(tl.expand_dims(lane,0),0)*precision))&((1<<precision)-1)
    return tl.convert_layout(tl.reshape(code,(K,TILE)),LAYOUT)
