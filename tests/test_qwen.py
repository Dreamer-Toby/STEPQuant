"""Optional integration checks against the pinned Transformers implementation."""
import torch
import pytest
transformers = pytest.importorskip('transformers')
if not hasattr(transformers, 'Qwen3_5ForCausalLM'):
    pytest.skip('requires the Qwen environment', allow_module_level=True)
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from transformers.models.qwen3_5 import modeling_qwen3_5 as native
from stepquant.adapters import ModelAdapter, RecurrentKernel
from stepquant.quantization import PackedState
from stepquant.calibration import calibrate


def test_native_recurrence_agreement():
    torch.manual_seed(12)
    q,k,v=torch.randn(1,9,2,8),torch.randn(1,9,2,8),torch.randn(1,9,2,6)
    g,b=-torch.rand(1,9,2),torch.rand(1,9,2)
    state=torch.randn(1,2,8,6)
    y,s=native.torch_recurrent_gated_delta_rule(q,k,v,g,b,state,True,True)
    y2,s2=RecurrentKernel('gdn')(q,k,v,g,b,state,True,True)
    torch.testing.assert_close(y,y2,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(s,s2,atol=1e-6,rtol=1e-5)


def test_small_model_calibrate_decode_and_reset(monkeypatch):
    monkeypatch.setattr(native, 'FusedRMSNormGated', None)
    torch.manual_seed(4)
    config=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,
        linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=8,
        linear_value_head_dim=8,layer_types=['linear_attention','full_attention'],
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.0,'mrope_section':[2,3,3]})
    model=Qwen3_5ForCausalLM(config).eval()
    ids=torch.tensor([[1,2,3,4,5,6,7,8]])
    with torch.inference_mode():
        adapter=ModelAdapter(model,collect=True,sample_every=2,max_snapshots=4)
        fp=model(ids,use_cache=True)
        assert fp.past_key_values.layers[0].recurrent_states.dtype==torch.float32
        artifact=calibrate(adapter.export_statistics(),pivots=1,value_group_size=4)
        adapter.close()
        adapter=ModelAdapter(model,artifact)
        quant=model(ids,use_cache=True)
        torch.testing.assert_close(fp.logits,quant.logits)  # prefill readouts aren't quantized
        assert isinstance(quant.past_key_values.layers[0].recurrent_states,PackedState)
        for _ in range(3):
            quant=model(torch.tensor([[9]]),past_key_values=quant.past_key_values,use_cache=True)
            assert torch.isfinite(quant.logits).all()
            assert isinstance(quant.past_key_values.layers[0].recurrent_states,PackedState)
        again=model(ids,use_cache=True)
        torch.testing.assert_close(fp.logits,again.logits)
        adapter.close()
