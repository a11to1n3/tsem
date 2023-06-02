import pytest
import torch
from tsem.model import TSEM

def test():
    series = torch.randn(1, 1, 256, 3)
    window_size = series.shape[-2]
    if window_size % 2 == 0:
        window_size -= 1

    tsem = TSEM(
        window_size = window_size,
        time_length = series.shape[-2],
        feature_length = series.shape[-1],
        n_classes = 1000
    )

    preds = tsem(series)
    assert preds.shape == (1, 1000), 'correct logits outputted'