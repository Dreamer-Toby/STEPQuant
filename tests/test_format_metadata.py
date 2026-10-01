"""The public registry contains the paper formats only."""
def test_paper_baseline_registry():
    import json
    from pathlib import Path
    import pytest
    from stepquant.formats import BASELINE_NAMES, FORMAT_NAMES, get_format
    assert set(BASELINE_NAMES) == {'fp32', 'sym_int4_g128', 'sym_int6_g128',
                                  'sym_int8_g128', 'dsq_int4', 'dsq_int6'}
    paths = list(Path('configs/formats').glob('*.json'))
    assert {p.stem for p in paths} == set(FORMAT_NAMES)
    for p in paths:
        assert json.loads(p.read_text()) == get_format(p.stem).to_dict()
    for name in ('fp16', 'bf16', 'fp8', 'damp', 'dsq_int8',
                 'hadamard_int6_g32', 'sym_int4_g32'):
        with pytest.raises(ValueError, match='unknown state format'):
            get_format(name)
