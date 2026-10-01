"""Byte carriers retain quantization levels but must not claim packed memory use."""
import pytest
import torch
pytest.importorskip("triton")
from stepquant.quantization import QuantizationPlan
from stepquant.kernels.pool import packed_layout


def test_byte_page_accounts_for_carriers_pivots_and_scales():
    bits=torch.tensor([[2,4,6,16],[16,16,16,16]])
    plan=QuantizationPlan('kda',bits,torch.ones_like(bits).float())
    offsets,rows,columns,total=packed_layout(plan,32,'byte')
    # First head: a complete 4x32 byte plane plus one pivot. Second: four pivots.
    payload=4*32+5*32*2
    assert offsets[8:]==[0,192]
    assert offsets[3]==128
    assert columns==[payload+3*2,-1]
    assert rows==[payload,payload+2,payload+4,-1,-1,-1,-1,-1]
    assert total==(payload+3*2+32*2+15)//16*16
    assert total>packed_layout(plan,32,'packed')[-1]
