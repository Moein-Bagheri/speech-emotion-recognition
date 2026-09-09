"""Plain and residual one-dimensional CNNs specified by the assignment."""

from __future__ import annotations

import torch
from torch import nn


class PlainConvBlock(nn.Module):
    """Conv1d + BatchNorm1d + ReLU + MaxPool1d."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class ResidualBlock1D(nn.Module):
    """A two-layer residual block followed by temporal max pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(inputs)
        outputs = self.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        outputs = self.relu(outputs + identity)
        return self.pool(outputs)


def downsample_lengths(lengths: torch.Tensor, num_pools: int = 3) -> torch.Tensor:
    """Propagate valid sequence lengths through stride-two max-pooling."""

    result = lengths
    for _ in range(num_pools):
        result = torch.div(result, 2, rounding_mode="floor")
    return result.clamp_min(1)


def temporal_average(
    feature_map: torch.Tensor,
    lengths: torch.Tensor | None,
) -> torch.Tensor:
    """Apply regular or length-masked global average pooling."""

    if lengths is None:
        return feature_map.mean(dim=-1)

    time = feature_map.shape[-1]
    positions = torch.arange(time, device=feature_map.device)
    mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(1).to(feature_map.dtype)
    summed = (feature_map * mask).sum(dim=-1)
    denominator = mask.sum(dim=-1).clamp_min(1.0)
    return summed / denominator


class _ClassifierBase(nn.Module):
    """Common pooling and two-linear-layer classifier head."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 4,
        feature_dim: int = 256,
        masked_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.masked_pooling = masked_pooling
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_classes),
        )

    def _pool(
        self,
        feature_map: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.masked_pooling:
            if lengths is None:
                raise ValueError(
                    "Valid lengths are required for masked global pooling."
                )
            lengths = downsample_lengths(lengths, num_pools=3)
        else:
            lengths = None
        return temporal_average(feature_map, lengths)


class Plain1DCNN(_ClassifierBase):
    """Three plain convolutional blocks and a two-layer classifier."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 4,
        feature_dim: int = 256,
        masked_pooling: bool = False,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            feature_dim=feature_dim,
            masked_pooling=masked_pooling,
        )
        self.encoder = nn.Sequential(
            PlainConvBlock(input_channels, 64),
            PlainConvBlock(64, 128),
            PlainConvBlock(128, feature_dim),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature_map = self.encoder(inputs)
        embedding = self._pool(feature_map, lengths)
        return self.classifier(embedding)


class Residual1DCNN(_ClassifierBase):
    """Three residual convolutional blocks and a two-layer classifier."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 4,
        feature_dim: int = 256,
        masked_pooling: bool = False,
    ) -> None:
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            feature_dim=feature_dim,
            masked_pooling=masked_pooling,
        )
        self.encoder = nn.Sequential(
            ResidualBlock1D(input_channels, 64),
            ResidualBlock1D(64, 128),
            ResidualBlock1D(128, feature_dim),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature_map = self.encoder(inputs)
        embedding = self._pool(feature_map, lengths)
        return self.classifier(embedding)


def build_model(
    architecture: str,
    input_channels: int,
    masked_pooling: bool = False,
) -> nn.Module:
    """Construct one of the two requested architectures."""

    normalized = architecture.strip().lower()
    if normalized == "plain":
        return Plain1DCNN(
            input_channels=input_channels,
            masked_pooling=masked_pooling,
        )
    if normalized == "residual":
        return Residual1DCNN(
            input_channels=input_channels,
            masked_pooling=masked_pooling,
        )
    raise ValueError("architecture must be 'plain' or 'residual'.")


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that receive gradients."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
