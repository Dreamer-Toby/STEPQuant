"""Compressed state-format pages with the standard SGLang pool API."""
import torch
from .pool import PackedStatePool
from .formats import codec, decode, codec_rows
from .fused import rows_kernel
from . import floating


def format_layout(plan, value_dim, spec):
    offsets, rows, columns = [], [], []
    cursor = 0
    for b in plan.bits.cpu().flatten().tolist():
        if b==16:cursor=(cursor+1)//2*2
        offsets.append(cursor)
        cursor += value_dim * b // 8
    cursor = (cursor + 3) // 4 * 4
    floating = spec.kind == 'fp32'
    dual = spec.kind == 'dsq'
    for b in plan.bits.cpu().flatten().tolist():
        rows.append(cursor if b!=16 and not floating else -1)
        cursor += 0 if floating or b==16 else (2 if dual else value_dim // spec.group_size * 4)
    cursor=(cursor+3)//4*4
    for _ in range(plan.bits.shape[0]):
        columns.append(cursor)
        cursor += value_dim * 4 if dual else 0
    return offsets, rows, columns, cursor


class FormatStatePool(PackedStatePool):
    def __init__(self, plan, slots, value_dim, spec, writeback_stream=None, layout=None):
        if writeback_stream is not None:
            raise ValueError('these fused formats require synchronous writeback')
        allowed=(spec.bits,)
        if not torch.isin(plan.bits,torch.tensor(allowed,device=plan.bits.device)).all():
            raise ValueError(f'precision map incompatible with {spec.name}')
        self.plan, self.spec = plan, spec
        self.heads, self.keys = plan.bits.shape
        self.values, self.device = value_dim, plan.bits.device
        if self.device.type != 'cuda':
            raise ValueError('fused format pool requires CUDA')
        if self.keys & (self.keys-1) or value_dim & (value_dim-1) or value_dim % 4:
            raise ValueError('key/value dimensions must be powers of two, value >= 4')
        if spec.kind == 'dsq' and self.keys > 128:
            raise ValueError('dual-axis g128 requires key_dim <=128')
        if spec.kind not in ('dsq', 'fp32') and value_dim % spec.group_size:
            raise ValueError('group size must divide value dimension')
        offsets, rows, columns, self.page_bytes = format_layout(plan, value_dim, spec) if layout is None else layout
        self.offsets = torch.tensor(offsets, device=self.device, dtype=torch.int32)
        self.rows = torch.tensor(rows, device=self.device, dtype=torch.int32)
        self.columns = torch.tensor(columns, device=self.device, dtype=torch.int32)
        self.bits = plan.bits.to(torch.int32).contiguous()
        self.impact = plan.impact.float().contiguous()
        self.data = torch.zeros(slots, self.page_bytes, device=self.device, dtype=torch.uint8)
        self.floating=spec.kind == "fp32"
        self.float_data=self.data.view(torch.float32) if self.floating else None
        self.pending = False
        self.dual=spec.kind == "dsq"
        self.tile=value_dim if spec.kind == "dsq" else min(value_dim,max(32,spec.group_size))
        width=spec.bits
        self.kw = dict(PAGE=self.page_bytes, H=self.heads, K=self.keys, V=value_dim,
                       GROUP=spec.group_size, KIND=spec.kind,
                        TILE=self.tile, WIDTH=width, num_warps=8 if self.tile>=128 else 4, enable_fp_fusion=False)

    def encode(self, state, slots):
        self.wait()
        if state.shape != (slots.numel(), self.heads, self.keys, self.values):
            raise ValueError('state shape mismatch')
        if slots.numel():
            if self.floating:
                self._float_codec(state.contiguous(),slots,True)
                return
            if self.dual:
                self._dual_writeback(state.contiguous(),slots)
                return
            self._group_codec(state.contiguous(),slots,True)

    def decode(self, slots):
        self.wait()
        state = torch.empty((slots.numel(), self.heads, self.keys, self.values), device=self.device, dtype=torch.float32)
        if slots.numel():
            if self.floating:self._float_codec(state,slots,False)
            elif not self.dual:self._group_codec(state,slots,False)
            else:codec[(slots.numel(), self.heads, self.values//self.tile)](*self._args(), slots, state, ENCODE=False, **self.kw)
        return state

    def step(self, q, k, v, g, beta, slots, *, normalize=False, scale=1., a_log=None, dt_bias=None):
        batch = slots.numel()
        out = torch.empty((batch, self.heads, self.values), device=self.device, dtype=q.dtype)
        if batch:
            q, k = q.reshape(batch, -1, self.keys), k.reshape(batch, -1, self.keys)
            v = v.reshape(batch, self.heads, self.values)
            if self.heads % q.shape[1] or any(t.stride(-1) != 1 for t in (q,k,v)):
                raise ValueError('invalid head layout')
            if self.floating:
                tile=min(32,self.values)
                floating.decode[(batch,self.heads,self.values//tile)](self.float_data,slots,q,k,v,g.contiguous(),beta.contiguous(),out,a_log,dt_bias,
                    H=self.heads,HK=q.shape[1],K=self.keys,V=self.values,KDA=self.plan.architecture=='kda',
                    NORMALIZE=normalize,RAW_GATES=a_log is not None,SCALE=scale,QS=q.stride(0),KS=k.stride(0),VS=v.stride(0),
                    TILE=tile,num_warps=4,enable_fp_fusion=False)
                return out
            temporary=torch.empty((batch,self.heads,self.keys,self.values),device=self.device,dtype=torch.float32) if self.dual else None
            decode[(batch, self.heads, self.values//self.tile)](*self._args(), slots, q,k,v,g.contiguous(),beta.contiguous(),out,a_log,dt_bias,
                HK=q.shape[1], KDA=self.plan.architecture=='kda', NORMALIZE=normalize, SCALE=scale,
                RAW_GATES=a_log is not None, QS=q.stride(0),KS=k.stride(0),VS=v.stride(0),
                State=temporary,UPDATE_ONLY=self.dual,**dict(self.kw,num_warps=16 if self.dual else self.kw["num_warps"]))
            if self.dual:self._dual_writeback(temporary,slots,rows_ready=True)
        return out


    def _dual_writeback(self,state,slots,rows_ready=False):
        if not rows_ready:
            rows_kernel[((self.heads*self.keys+7)//8,slots.numel())](state,self.data,self.bits,self.impact,self.offsets,
                self.rows,slots,PAGE=self.page_bytes,H=self.heads,K=self.keys,V=self.values,GROUP=self.values,
                KDA=True,ROWS=8,DSQ=self.spec.kind=='dsq',num_warps=4,enable_fp_fusion=False)
        tile=min(16,self.values)
        kw=dict(self.kw,TILE=tile,num_warps=4)
        codec[(slots.numel(),self.heads,self.values//tile)](*self._args(),slots,state,ENCODE=True,ROW_READY=True,**kw)


    def _float_codec(self,state,slots,encode):
        n=self.heads*self.keys*self.values
        floating.codec[(slots.numel(),(n+1023)//1024)](self.float_data,slots,state,N=n,ENCODE=encode,num_warps=4)


    def _group_codec(self,state,slots,encode):
        rows=min(self.keys,max(1,1024//self.tile))
        codec_rows[(self.heads*self.keys//rows,slots.numel(),self.values//self.tile)](*self._args(),slots,state,
            ENCODE=encode,ROW_BLOCK=rows,**dict(self.kw,num_warps=4))
