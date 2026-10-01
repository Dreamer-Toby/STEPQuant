"""B512 TP logits: gather shards, then write the final buffer once."""
import torch
import triton
import triton.language as tl

@triton.jit
def arrange_logits(Source,Output,ROWS:tl.constexpr,LOCAL:tl.constexpr,VOCAB:tl.constexpr,STRIDE:tl.constexpr,BLOCK:tl.constexpr):
    index=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    row=index//VOCAB;column=index%VOCAB
    source_index=((column//LOCAL)*ROWS+row)*LOCAL+column%LOCAL
    value=tl.load(Source+source_index,row<ROWS,0)
    tl.store(Output+row*STRIDE+column,value,row<ROWS)

def get_logits(original,self,hidden_states,lm_head,logits_metadata,embedding_bias=None):
    if (hidden_states.shape[0]!=512 or not hidden_states.is_cuda
        or not self.do_tensor_parallel_all_gather or self.use_attn_tp_group
        or self.do_tensor_parallel_all_gather_dp_attn or self.final_logit_softcapping):
        return original(self,hidden_states,lm_head,logits_metadata,embedding_bias)
    from sglang.srt.distributed import get_tp_group
    local=self._compute_lm_head(hidden_states,lm_head,embedding_bias)
    if self.logit_scale is not None:local.mul_(self.logit_scale)
    group=get_tp_group();rows,shard=local.shape
    gathered=torch.empty((group.world_size*rows,shard),device=local.device,dtype=local.dtype)
    group.all_gather_into_tensor(gathered,local)
    out=logits_metadata.next_token_logits_buffer
    if out is None:out=torch.empty((rows,self.vocab_size),device=local.device,dtype=torch.float32)
    assert out.dtype==torch.float32 and out.shape==(rows,self.vocab_size) and out.stride(1)==1
    arrange_logits[(triton.cdiv(rows*self.vocab_size,2048),)](gathered,out,rows,shard,self.vocab_size,out.stride(0),2048)
    return out
