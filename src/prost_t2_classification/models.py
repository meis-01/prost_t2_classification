from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


class RealAmplitudeCNN(nn.Module):
    def __init__(self, in_channels: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _real_block(in_channels, 32),
            nn.MaxPool2d(2),
            _real_block(32, 64),
            nn.MaxPool2d(2),
            _real_block(64, 128),
            nn.MaxPool2d(2),
            _real_block(128, 192),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(-1)


def _real_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.real_weight = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.imag_weight = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        real = self.real_weight(x.real) - self.imag_weight(x.imag)
        imag = self.real_weight(x.imag) + self.imag_weight(x.real)
        return torch.complex(real, imag)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.real_norm = nn.BatchNorm2d(channels)
        self.imag_norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.real_norm(x.real), self.imag_norm(x.imag))


class ModReLU(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.full((channels,), -0.1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(x)
        bias = self.bias.view(1, -1, 1, 1)
        scale = F.relu(magnitude + bias) / (magnitude + self.eps)
        return x * scale


class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(
            F.avg_pool2d(x.real, self.kernel_size),
            F.avg_pool2d(x.imag, self.kernel_size),
        )


class ComplexBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels)
        self.norm1 = ComplexBatchNorm2d(out_channels)
        self.act1 = ModReLU(out_channels)
        self.conv2 = ComplexConv2d(out_channels, out_channels)
        self.norm2 = ComplexBatchNorm2d(out_channels)
        self.act2 = ModReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class ComplexT2CNN(nn.Module):
    def __init__(self, in_channels: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        self.block1 = ComplexBlock(in_channels, 32)
        self.pool1 = ComplexAvgPool2d(2)
        self.block2 = ComplexBlock(32, 64)
        self.pool2 = ComplexAvgPool2d(2)
        self.block3 = ComplexBlock(64, 128)
        self.pool3 = ComplexAvgPool2d(2)
        self.block4 = ComplexBlock(128, 192)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(192, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        magnitude_features = torch.abs(x)
        pooled = F.adaptive_avg_pool2d(magnitude_features, 1).flatten(1)
        return self.classifier(self.dropout(pooled)).squeeze(-1)


def build_model(mode: Literal["real", "complex"], *, in_channels: int = 5, dropout: float = 0.2) -> nn.Module:
    if mode == "real":
        return RealAmplitudeCNN(in_channels=in_channels, dropout=dropout)
    if mode == "complex":
        return ComplexT2CNN(in_channels=in_channels, dropout=dropout)
    raise ValueError(f"Unknown model mode: {mode}")
