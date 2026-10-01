"""Fixed-layout GPU pages, shared calibration metadata, and request-slot operations."""
import torch
from .readout import readout
from .segment_read import readout as segment_read, readout_kda
from .fit import fit
from .rows import prepare_rows_kernel
from .byte_codec import byte_decode
from .fused import codec_kernel, update_kernel, column_kernel, rows_kernel, prepare_inputs_kernel


def _capture_id(stream):
    from cuda.bindings import runtime
    status,_,identifier,*_=runtime.cudaStreamGetCaptureInfo(stream.cuda_stream)
    if status != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f'cannot query CUDA capture: {status}')
    return identifier


def packed_layout(plan,value_dim,storage='packed'):
    if storage not in ('packed','byte'):
        raise ValueError('state storage must be packed or byte')
    bits=plan.bits.cpu().tolist()
    offsets,rows,columns=[],[],[]
    cursor=0
    # Four-value tiles coalesce mixed KDA key rows. Only FP16 pivots and
    # tile strides need half alignment; integer codes remain tightly packed.
    tile=min(4 if plan.architecture=='kda' else 16,value_dim)
    strides=[]
    if storage=='byte':
        for head in bits:
            code_base=cursor
            if any(b!=16 for b in head):cursor+=len(head)*value_dim
            pivot_count=sum(b==16 for b in head)
            pivot_base=cursor
            pivot_index=0
            for i,b in enumerate(head):
                if b==16:
                    offsets.append(pivot_base+2*pivot_index*value_dim)
                    pivot_index+=1
                else:
                    offsets.append(code_base+i)
            cursor+=2*pivot_count*value_dim
            strides.append(code_base)
    else:
        for head in bits:
            local=0
            for b in head:
                if b==16:local=(local+1)//2*2
                offsets.append(cursor+local)
                local+=tile*b//8
            stride=(local+1)//2*2
            strides.append(stride)
            cursor+=stride*(value_dim//tile)
    offsets.extend(strides)
    cursor=(cursor+1)//2*2
    for head in bits:
        for b in head:
            rows.append(cursor if b!=16 else -1)
            if b!=16:
                cursor+=2*(value_dim//plan.value_group_size if b==2 and plan.architecture=='gdn' else 1)
    for head in bits:
        columns.append(cursor if any(b!=16 for b in head) else -1)
        if any(b!=16 for b in head):
            cursor+=2*value_dim
    if storage=='byte':cursor=(cursor+15)//16*16
    return offsets,rows,columns,cursor


class PackedStatePool:
    def __init__(self, plan, slots, value_dim=128, writeback_stream=None, writeback_workspace=None, writeback_chunk=64,storage='packed'):
        if plan.bits.device.type != 'cuda':
            raise ValueError('Triton state pool requires CUDA')
        if storage not in ('packed','byte'):
            raise ValueError('state storage must be packed or byte')
        if storage=='byte' and writeback_stream is None:
            raise ValueError('byte storage requires a writeback stream')
        self.storage=storage
        self.early_writeback=True
        self.plan=plan
        self.heads,self.keys=plan.bits.shape
        self.values=value_dim
        if self.keys & (self.keys-1) or value_dim & (value_dim-1) or value_dim%4:
            raise ValueError('key/value dimensions must be powers of two, value_dim >= 4')
        if plan.architecture=='gdn' and value_dim%plan.value_group_size:
            raise ValueError('value dimension must be divisible by value_group_size')
        offsets,rows,columns,cursor=packed_layout(plan,value_dim,storage)
        self.page_bytes=cursor
        self.device=plan.bits.device
        self.offsets=torch.tensor(offsets,device=self.device,dtype=torch.int32)
        self.rows=torch.tensor(rows,device=self.device,dtype=torch.int32)
        self.columns=torch.tensor(columns,device=self.device,dtype=torch.int32)
        self.data=torch.zeros(slots,cursor,device=self.device,dtype=torch.uint8)
        self.bits=plan.bits.to(torch.int32).contiguous()
        self.head_groups=[]
        if plan.architecture=='gdn':
            widths=plan.bits[:,0].cpu().tolist()
            for width in sorted(set(widths)):
                ids=[i for i,b in enumerate(widths) if b==width]
                self.head_groups.append((width,torch.tensor(ids,device=self.device,dtype=torch.int32)))
        self.read_groups=[]
        if self.head_groups:
            integer_ids=[i for i,b in enumerate(widths) if b!=16]
            if integer_ids:
                self.read_groups.append((0,torch.tensor(integer_ids,device=self.device,dtype=torch.int32)))
            self.read_groups.extend((width,ids) for width,ids in self.head_groups if width==16)
        self.impact=plan.impact.float().contiguous()
        # Bound the fit denominator before selecting the approximate division.
        # r <= 65504, code normalization <= 64 and integer levels <= 127.
        # General float32 division remains available for extreme custom weights.
        max_impact=float(self.impact.max().item())
        self.fast_fit=self.keys*(65504.*64*127*max_impact)**2 < 2.**124
        self.coefficients=None
        self.row_tile=min(16 if storage=='byte' else 32,value_dim,plan.value_group_size) if plan.architecture=='gdn' else min(16 if storage=='byte' else 8,value_dim)
        self._init_writeback(writeback_stream,writeback_chunk)
        groups=value_dim//(plan.value_group_size if plan.architecture=='gdn' else value_dim)
        self._init_workspace(slots,writeback_workspace,groups)
        self.kw=dict(PAGE=cursor,H=self.heads,K=self.keys,V=value_dim,
                     GROUP=plan.value_group_size if plan.architecture=='gdn' else value_dim,
                     KDA=plan.architecture=='kda',num_warps=8,enable_fp_fusion=False)

    def _init_writeback(self,writeback_stream,writeback_chunk):
        if writeback_chunk < 0:
            raise ValueError("writeback chunk must be nonnegative")
        self.writeback_chunk=writeback_chunk
        self.writeback_stream=writeback_stream
        self.pending=False
        self.captured_consumers=False
        self.consumer_dependencies=set()
        self.ready_capture=None
        self.deferred=None
        self.ready=torch.cuda.Event() if writeback_stream is not None else None
        self.graph_ready=torch.cuda.Event(external=True) if writeback_stream is not None else None
        self.completion=self.ready
        if self.graph_ready is not None:
            self.graph_ready.record(torch.cuda.current_stream(self.device))

    def _init_workspace(self,slots,writeback_workspace,groups=1,partial_components=1,coefficient_planes=3):
        writeback_stream=self.writeback_stream
        self.scratch=None
        if writeback_stream is not None:
            # Writer inputs must live outside CUDA Graph's private allocator:
            # a graph may reuse an earlier layer's temporary for a later tensor.
            # The next replay can overwrite that alias before the later layer waits.
            self.scratch=(
                torch.empty((slots,self.heads,self.keys),device=self.device,dtype=torch.float32),
                torch.empty((slots,self.heads,self.keys),device=self.device,dtype=torch.float32),
                torch.empty((slots,self.heads,self.values),device=self.device,dtype=torch.float32),
                torch.empty((slots,self.heads,self.keys,partial_components*self.values//self.row_tile),device=self.device,dtype=torch.float32),
                torch.empty(slots,device=self.device,dtype=torch.int64))
            for tensor in self.scratch:
                tensor.record_stream(writeback_stream)
            # Serial background work shares row coefficients across layers.
            workspace={} if writeback_workspace is None else writeback_workspace
            key=(self.device.index,writeback_stream.cuda_stream,slots,self.heads,self.keys,groups,coefficient_planes)
            if key not in workspace:
                workspace[key]=torch.empty((slots,coefficient_planes,self.heads,groups,self.keys),device=self.device,dtype=torch.float32)
            self.coefficients=workspace[key]
            self.coefficients.record_stream(writeback_stream)

    @property
    def nbytes(self):
        return self.data.numel()

    def configure_writeback(self, stream, chunk):
        """Bind a capture's writer without replacing state or scratch storage.

        Captured graphs retain their original stream. Completion dependencies
        order shared pages/coefficient scratch when a different graph replays.
        """
        if self.writeback_stream is None or chunk < 0:
            raise ValueError('configuring writeback requires an asynchronous pool and nonnegative chunk')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('select writeback resources before CUDA graph capture')
        self.wait()
        if stream != self.writeback_stream:
            for tensor in (*self.scratch, self.coefficients):
                tensor.record_stream(stream)
            self.writeback_stream = stream
        self.writeback_chunk = chunk

    def _args(self):
        return self.data,self.bits,self.impact,self.offsets,self.rows,self.columns

    def encode(self,state,slots):
        self.wait()
        if state.shape != (slots.numel(),self.heads,self.keys,self.values):
            raise ValueError('state must be [active slots, heads, key, value]')
        if slots.numel():
            state=state.contiguous()
            self._writeback(state,slots)

    def decode(self,slots):
        self.wait()
        state=torch.empty((slots.numel(),self.heads,self.keys,self.values),device=self.device,dtype=torch.float32)
        if slots.numel():
            kernel=byte_decode if self.storage=='byte' else codec_kernel
            kernel[(slots.numel(),self.heads)](self.data,self.bits,self.offsets,self.rows,self.columns,slots,state,**self.kw)
        return state

    def step(self,q,k,v,g,beta,slots,*,normalize=False,scale=1.,a_log=None,dt_bias=None):
        self.wait()
        if self.deferred is not None and not self.deferred.attention_windows:
            self.deferred.release_window()
        batch=slots.numel()
        if not batch:
            return torch.empty((0,self.heads,self.values),device=q.device,dtype=q.dtype)
        q=q.reshape(batch,-1,self.keys)
        k=k.reshape(batch,-1,self.keys)
        v=v.reshape(batch,self.heads,self.values)
        if q.stride(-1)!=1 or k.stride(-1)!=1 or v.stride(-1)!=1:
            raise ValueError("head coordinates must be contiguous")
        hq=q.shape[1]
        if self.heads%hq:
            raise ValueError('value heads must be divisible by query heads')
        out=torch.empty((batch,self.heads,self.values),device=q.device,dtype=q.dtype)
        asynchronous=self.writeback_stream is not None
        tile=self.row_tile
        raw_inputs=asynchronous and (self.plan.architecture=="gdn" or (self.storage=="byte" and batch<=64))
        segmented=asynchronous and self.plan.architecture=="gdn" and (
            (self.storage=="byte" and batch>=128) or
            (self.storage=="packed" and batch>=64 and self.keys>=4 and self.values>=16))
        kda_segments=asynchronous and self.plan.architecture=='kda' and self.storage=='packed' and batch>=256 and self.keys>=4 and self.values>=32
        if segmented:
            tile=min(32,self.values,self.plan.value_group_size)
        elif kda_segments:
            tile=min(32 if batch<512 else 64,self.values)
        if asynchronous:
            temporary=None
            if batch>self.scratch[0].shape[0]:
                raise ValueError('active batch exceeds the state pool workspace capacity')
            update_key,decay,delta,partial,stable_slots=(tensor[:batch] for tensor in self.scratch)
            if segmented or kda_segments:
                partial=partial.reshape(-1)[:batch*self.heads*self.keys*(self.values//tile)].view(batch,self.heads,self.keys,self.values//tile)
            if raw_inputs:
                g,beta=g.contiguous(),beta.contiguous()
            else:
                prepared_q=torch.empty_like(update_key)
                prepared_beta=torch.empty((batch,self.heads),device=self.device,dtype=torch.float32)
                prepare_inputs_kernel[(self.heads,batch)](q,k,g.contiguous(),beta.contiguous(),a_log,dt_bias,
                    prepared_q,update_key,decay,prepared_beta,H=self.heads,HK=hq,K=self.keys,
                    KDA=self.plan.architecture=='kda',QS=q.stride(0),KS=k.stride(0),NORMALIZE=normalize,
                    RAW_GATES=a_log is not None,SCALE=scale,num_warps=1,enable_fp_fusion=True)
                q,k,g,beta=prepared_q,update_key,decay,prepared_beta
                hq=self.heads;normalize=False;scale=1.;a_log=dt_bias=None
        else:
            temporary=torch.empty((batch,self.heads,self.keys,self.values),device=self.device,dtype=torch.float32)
            partial=torch.empty((batch,self.heads,self.keys,self.values//tile),device=self.device,dtype=torch.float32)
            update_key=decay=delta=None
        kw={k:v for k,v in self.kw.items() if k!='num_warps'}
        if asynchronous:
            kw['enable_fp_fusion']=True
        groups=(self.read_groups if asynchronous else self.head_groups) if self.head_groups and batch>=16 else [(0,None)]
        for width,head_ids in groups:
            count=self.heads if head_ids is None else head_ids.numel()
            if kda_segments:
                readout_kda[(batch,count,self.values//tile)](self.data,self.bits,self.offsets,self.rows,self.columns,
                    slots,q,k,v,g,beta,out,delta,partial,stable_slots,PAGE=self.page_bytes,H=self.heads,K=self.keys,
                    V=self.values,VS=v.stride(0),TILE=tile,num_warps=1,maxnreg=96,enable_fp_fusion=True)
            elif asynchronous:
                read_tile=self.row_tile if segmented and width==16 else tile
                use_segments=segmented and width!=16
                reader=segment_read if use_segments else readout
                warps=1 if self.storage=="byte" or use_segments or self.plan.architecture=="kda" else 4
                read_kw=dict(KEY_LANES=16 if batch<=128 else 32,PART=4 if self.storage=='packed' and batch>=256 else 8) if use_segments else {}
                grid=(batch,count,self.values//read_tile) if self.plan.architecture=='kda' and self.storage=='packed' else (self.values//read_tile,count,batch)
                reader[grid](self.data,self.bits,self.offsets,self.rows,self.columns,
                    slots,q,k,v,g,beta,out,delta,partial,VS=v.stride(0),TILE=read_tile,
                    HeadIds=head_ids,WIDTH=width,NW=warps,num_warps=warps,**read_kw,
                    StableSlots=stable_slots,RawKey=update_key if raw_inputs else None,RawDecay=decay if raw_inputs else None,
                    ALog=a_log,Bias=dt_bias,HK=hq,QS=q.stride(0),KS=k.stride(0),NORMALIZE=normalize,SCALE=scale,
                    INTEGER_HEADS=bool(width==0 and head_ids is not None),BYTE=self.storage=='byte',**kw)
            else:
                update_kernel[(self.values//tile,count,batch)](self.data,self.bits,self.offsets,self.rows,self.columns,
                    slots,q,k,v,g.contiguous(),beta.contiguous(),out,temporary,a_log,dt_bias,
                    HK=hq,NORMALIZE=normalize,SCALE=scale,RAW_GATES=a_log is not None,
                    QS=q.stride(0),KS=k.stride(0),VS=v.stride(0),TILE=tile,num_warps=4,
                    Partial=partial,HeadIds=head_ids,WIDTH=width,**kw)
        if not asynchronous:
            self._writeback(temporary,slots,rows_ready=True,partial=partial,row_tile=tile)
        else:
            self._schedule_writeback(update_key,decay,delta,partial,stable_slots)
        return out

    def _schedule_writeback(self,update_key,decay,delta,partial,stable_slots):
        if self.deferred is not None:
            self.deferred.append((self,update_key,decay,delta,partial,stable_slots))
            return
        self.writeback_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.writeback_stream):
            self._finish_update(update_key,decay,delta,partial,stable_slots)
            self.graph_ready.record()
            self.ready.record()
        for tensor in (update_key,decay,delta,partial,stable_slots):
            tensor.record_stream(self.writeback_stream)
        self.ready_capture=_capture_id(torch.cuda.current_stream(self.device)) if torch.cuda.is_current_stream_capturing() else None
        self.completion=self.graph_ready if self.ready_capture is not None else self.ready
        self.pending=True

    def _finish_update(self,update_key,decay,delta,partial,slots):
        coefficients=self.coefficients[:slots.numel()]
        row_tile=self.values//partial.shape[-1]
        kw={k:v for k,v in self.kw.items() if k!='num_warps'}
        kw['enable_fp_fusion']=True
        prepare_rows_kernel[(self.heads,slots.numel())](self.data,self.bits,self.impact,self.rows,
            slots,update_key,decay,partial,coefficients,ROWTILE=row_tile,NW=2,num_warps=2,**kw)
        small_writer=self.plan.architecture=='kda' and (self.storage=='byte' or slots.numel()>=64)
        tile=min(8 if small_writer else 16,self.values)
        warps=1 if small_writer or (self.storage=='packed' and self.plan.architecture=='gdn') else 4
        groups=self.head_groups if self.head_groups and self.storage=="packed" else [(0,None)]
        for width,head_ids in groups:
            if width==16:
                continue
            count=self.heads if head_ids is None else head_ids.numel()
            fit[(self.values//tile,count,slots.numel())](self.data,self.bits,self.offsets,
                self.columns,slots,coefficients,delta,TILE=tile,NW=warps,num_warps=warps,
                HeadIds=head_ids,WIDTH=width,FAST_RATIO=self.fast_fit,BYTE=self.storage=='byte',SPECIALIZE=self.storage=='byte' and self.plan.architecture=='gdn',**kw)


    def _writeback(self,state,slots,rows_ready=False,partial=None,row_tile=32):
        kw={k:v for k,v in self.kw.items() if k!='num_warps'}
        if not rows_ready:
            rows_kernel[((self.heads*self.keys+7)//8,slots.numel())](state,self.data,self.bits,self.impact,self.offsets,
                self.rows,slots,ROWS=8,TILED=self.storage=='packed',num_warps=4,**kw)
        tile=min(16,self.values)
        if self.storage=='byte':
            fit[(self.values//tile,self.heads,slots.numel())](self.data,self.bits,self.offsets,
                self.columns,slots,None,None,TILE=tile,NW=4,num_warps=4,BYTE=True,
                State=state,Rows=self.rows,Impact=self.impact,**kw)
        else:
            column_kernel[(self.values//tile,self.heads,slots.numel())](state,*self._args(),slots,
                TILE=tile,num_warps=4,SPECIALIZE=slots.numel()>=16,Partial=partial,ROW_PARTIALS=partial is not None,ROWTILE=row_tile,**kw)


    def wait(self):
        stream=torch.cuda.current_stream(self.device)
        if getattr(self,'deferred',None) is not None:
            if not self.deferred.joined:
                stream.wait_event(self.graph_ready)
        elif self.pending:
            event=self.completion
            if torch.cuda.is_current_stream_capturing():
                # Internal events join only their own capture. Previous captures
                # and eager submissions require explicit external event nodes.
                if self.ready_capture==_capture_id(stream):
                    event=self.ready
                else:
                    self.captured_consumers=True
                    # Future producers may use another graph or eager execution.
                    # Retain the initial event too: CUDA graph nodes do not own
                    # external events after the producer graph is destroyed.
                    if self.completion is not self.ready and self.completion is not self.graph_ready:
                        self.consumer_dependencies.add(self.completion)
                        stream.wait_event(self.completion)
                    event=self.graph_ready
            stream.wait_event(event)
            # A wait orders this stream only; other consumers still need it.

    def clear(self,slots):
        self.wait()
        self.data[slots]=0

    def copy(self,source,destination):
        self.wait()
        # Advanced indexing creates a temporary: overlapping copy is well-defined.
        self.data[destination]=self.data[source]
