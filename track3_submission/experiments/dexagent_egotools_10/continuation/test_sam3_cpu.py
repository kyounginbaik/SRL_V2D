"""Reproduce the CPU cache and MLP checks for the pinned SAM3 patch."""
import torch
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.vitdet import Mlp

for resolution in [56,112,224]:
    cached=PositionEmbeddingSine(256,normalize=True,precompute_resolution=resolution)
    plain=PositionEmbeddingSine(256,normalize=True)
    for size in [resolution//14,resolution//7,resolution//4]:
        value=torch.zeros(2,1,size,size)
        assert torch.equal(cached(value),plain(value))
    print(f'[PASS] cached position encoding equals uncached at {resolution}',flush=True)

for size,tokens,hidden in [(8,3,16),(32,17,64),(64,31,128)]:
    torch.manual_seed(0)
    mlp=Mlp(size,hidden_features=hidden).eval()
    value=torch.randn(2,tokens,size)
    with torch.inference_mode():
        expected=mlp.fc2(mlp.act(mlp.fc1(value)))
        actual=mlp(value)
    assert actual.dtype==expected.dtype==torch.float32
    assert torch.equal(actual,expected)
    print(f'[PASS] CPU MLP {tuple(value.shape)} equals Linear-GELU-Linear',flush=True)
