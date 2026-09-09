from __future__ import annotations

import pytest
import torch

from ser.models import (
    Plain1DCNN,
    Residual1DCNN,
    downsample_lengths,
)


@pytest.mark.parametrize(
    ("model_class", "channels"),
    [
        (Plain1DCNN, 40),
        (Plain1DCNN, 120),
        (Residual1DCNN, 40),
        (Residual1DCNN, 120),
    ],
)
def test_models_return_four_logits(model_class, channels: int) -> None:
    model = model_class(input_channels=channels)
    inputs = torch.randn(2, channels, 300)
    outputs = model(inputs)
    assert outputs.shape == (2, 4)
    assert torch.isfinite(outputs).all()


def test_masked_pooling_accepts_lengths() -> None:
    model = Plain1DCNN(input_channels=40, masked_pooling=True)
    inputs = torch.randn(2, 40, 300)
    lengths = torch.tensor([300, 94])
    outputs = model(inputs, lengths)
    assert outputs.shape == (2, 4)


def test_lengths_follow_three_pooling_layers() -> None:
    lengths = torch.tensor([300, 299, 94])
    result = downsample_lengths(lengths)
    assert result.tolist() == [37, 37, 11]
