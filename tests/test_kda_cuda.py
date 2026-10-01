"""Run with the Kimi environment and a CUDA device; otherwise explicitly skipped."""
import pytest
import torch
fla = pytest.importorskip('fla.ops.kda')
from stepquant.adapters import RecurrentKernel


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required for native FLA')
def test_native_kda_coordinates_and_normalization():
    torch.manual_seed(17)
    q,k,v=[torch.randn(1,8,2,16,device='cuda',dtype=torch.float32) for _ in range(3)]
    g=-torch.rand_like(q)
    beta=torch.rand(1,8,2,device='cuda')
    state=torch.randn(1,2,16,16,device='cuda')
    expected,final=fla.fused_recurrent_kda(q,k,v,g,beta,initial_state=state.clone(),
        output_final_state=True,use_qk_l2norm_in_kernel=True)
    actual,state2=RecurrentKernel('kda')(q,k,v,g,beta,initial_state=state.clone(),
        output_final_state=True,use_qk_l2norm_in_kernel=True)
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
    torch.testing.assert_close(state2,final,atol=2e-5,rtol=2e-4)
