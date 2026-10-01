"""Optional actual Kimi checkpoint-code integration using a tiny random configuration."""
import os
from unittest.mock import patch
import torch
import pytest
from stepquant.adapters import ModelAdapter
from stepquant.calibration import calibrate
from stepquant.quantization import PackedState


@pytest.mark.skipif(not os.environ.get('STEPQUANT_KIMI_CHECKPOINT') or not torch.cuda.is_available(),
                    reason='set STEPQUANT_KIMI_CHECKPOINT and use Kimi CUDA environment')
def test_kimi_model_with_accelerate_wrapped_forward():
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate.hooks import add_hook_to_module, AlignDevicesHook
    path=os.environ['STEPQUANT_KIMI_CHECKPOINT']
    config=AutoConfig.from_pretrained(path,trust_remote_code=True,local_files_only=True)
    for k,v in dict(vocab_size=128,pad_token_id=0,hidden_size=64,intermediate_size=128,
                    moe_intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,
                    num_key_value_heads=2,num_experts=4,num_experts_per_token=2,
                    kv_lora_rank=16,qk_nope_head_dim=16,qk_rope_head_dim=16,v_head_dim=16).items():
        setattr(config,k,v)
    config.linear_attn_config=dict(full_attn_layers=[2],kda_layers=[1],head_dim=16,num_heads=4,short_conv_kernel_size=4)
    config._attn_implementation='eager'
    with patch('transformers.utils.auto_docstring',lambda fn:fn):
        model=AutoModelForCausalLM.from_config(config,trust_remote_code=True).cuda().bfloat16().eval()
    model.config._attn_implementation='eager'
    model.model._use_flash_attention_2=False
    attn=next(m for m in model.modules() if type(m).__name__=='KimiDeltaAttention')
    with torch.no_grad():
        attn.dt_bias.zero_()
        attn.A_log.zero_()
    add_hook_to_module(attn,AlignDevicesHook(execution_device=torch.device('cuda:0')))
    with torch.inference_mode():
        adapter=ModelAdapter(model,collect=True,sample_every=2,max_snapshots=4)
        ids=torch.tensor([[1,2,3,4,5,6,7,8]],device='cuda')
        output=model(ids,use_cache=True,generation_mode=True)
        artifact=calibrate(adapter.export_statistics(),pivots=1,value_group_size=4)
        adapter.close()
        adapter=ModelAdapter(model,artifact)
        quant=model(ids,use_cache=True,generation_mode=True)
        torch.testing.assert_close(quant.logits,output.logits)
        assert isinstance(quant.past_key_values.recurrent_states[0],PackedState)
        quant=model(torch.tensor([[9]],device='cuda'),past_key_values=quant.past_key_values,use_cache=True,generation_mode=True)
        assert isinstance(quant.past_key_values.recurrent_states[0],PackedState)
        assert torch.isfinite(quant.logits).all()
        adapter.close()
