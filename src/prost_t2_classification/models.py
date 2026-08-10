from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


COMPLEX_CHANNELS: tuple[int, int, int, int] = (32, 64, 128, 192)
# sqrt(2) times the complex widths. A real convolution at these widths has
# approximately the same number of scalar weights as a complex convolution.
PARAMETER_MATCHED_REAL_CHANNELS: tuple[int, int, int, int] = (45, 91, 181, 272)


class RealAmplitudeCNN(nn.Module):
    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        c1, c2, c3, c4 = PARAMETER_MATCHED_REAL_CHANNELS
        self.features = nn.Sequential(
            _real_block(in_channels, c1),
            nn.MaxPool2d(2),
            _real_block(c1, c2),
            nn.MaxPool2d(2),
            _real_block(c2, c3),
            nn.MaxPool2d(2),
            _real_block(c3, c4),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(c4, 1),
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
        with torch.no_grad():
            self.real_weight.weight.mul_(1 / math.sqrt(2))
            self.imag_weight.weight.mul_(1 / math.sqrt(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        real = self.real_weight(x.real) - self.imag_weight(x.imag)
        imag = self.real_weight(x.imag) + self.imag_weight(x.real)
        return torch.complex(real, imag)


class ComplexRMSNorm2d(nn.Module):
    """Phase-equivariant channel normalization using complex RMS power."""

    def __init__(self, channels: int, *, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_power", torch.ones(channels))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            power = (x.real.square() + x.imag.square()).mean(dim=(0, 2, 3))
            with torch.no_grad():
                self.running_power.lerp_(power.detach(), self.momentum)
        else:
            power = self.running_power
        scale = torch.exp(self.log_scale) * torch.rsqrt(power + self.eps)
        return x * scale.view(1, -1, 1, 1)


class ModReLU(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(x)
        bias = self.bias.view(1, -1, 1, 1)
        scale = F.relu(magnitude + bias) / (magnitude + self.eps)
        return x * scale


class ComplexMagnitudeMaxPool2d(nn.Module):
    """Select the full complex value with maximum amplitude per pooling window."""

    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("ComplexMagnitudeMaxPool2d expects a complex tensor")

        amplitude_squared = x.real.square() + x.imag.square()
        _, indices = F.max_pool2d(
            amplitude_squared,
            self.kernel_size,
            stride=self.stride,
            return_indices=True,
        )
        output_shape = indices.shape
        flat_indices = indices.flatten(start_dim=2)
        pooled_real = torch.gather(x.real.flatten(start_dim=2), 2, flat_indices).reshape(output_shape)
        pooled_imag = torch.gather(x.imag.flatten(start_dim=2), 2, flat_indices).reshape(output_shape)
        return torch.complex(pooled_real, pooled_imag)


class ComplexBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels)
        self.norm1 = ComplexRMSNorm2d(out_channels)
        self.act1 = ModReLU(out_channels)
        self.conv2 = ComplexConv2d(out_channels, out_channels)
        self.norm2 = ComplexRMSNorm2d(out_channels)
        self.act2 = ModReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class ComplexT2CNN(nn.Module):
    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        c1, c2, c3, c4 = COMPLEX_CHANNELS
        self.block1 = ComplexBlock(in_channels, c1)
        self.pool1 = ComplexMagnitudeMaxPool2d(2)
        self.block2 = ComplexBlock(c1, c2)
        self.pool2 = ComplexMagnitudeMaxPool2d(2)
        self.block3 = ComplexBlock(c2, c3)
        self.pool3 = ComplexMagnitudeMaxPool2d(2)
        self.block4 = ComplexBlock(c3, c4)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(c4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        pooled = F.adaptive_avg_pool2d(torch.abs(x), 1).flatten(1)
        return self.classifier(self.dropout(pooled)).squeeze(-1)


def build_model(
    mode: Literal["real", "complex"],
    *,
    in_channels: int = 4,
) -> nn.Module:
    if mode == "real":
        return RealAmplitudeCNN(in_channels=in_channels)
    if mode == "complex":
        return ComplexT2CNN(in_channels=in_channels)
    raise ValueError(f"Unknown model mode: {mode}")
