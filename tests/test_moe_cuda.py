"""SGLang actually finds the generated device-matched tuning files."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
pytest.importorskip('sglang')
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('variant',['sglang','sglang_shared_prompt'])
@pytest.mark.parametrize('down',[False,True])
def test_sglang_matching_configuration(tmp_path,monkeypatch,variant,down):
    from stepquant.moe import materialize_configs
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config as module
    source=Path('configs/kernels')/variant
    generated=materialize_configs(source,output_root=tmp_path)
    if generated is None:pytest.skip('no template for this device family')
    monkeypatch.setenv('SGLANG_MOE_CONFIG_DIR',str(generated))
    monkeypatch.setattr(module,'get_global_server_args',lambda:SimpleNamespace(enable_deterministic_inference=False))
    module.get_moe_configs.cache_clear()
    actual=module.get_moe_configs(256,256,None,down_moe=down)
    template=next(p for p in source.rglob('*.json') if p.stem.endswith('_down')==down)
    expected={int(k):v for k,v in json.loads(template.read_text()).items()}
    assert actual==expected
    module.get_moe_configs.cache_clear()
