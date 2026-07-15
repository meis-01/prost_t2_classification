from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


ComplexActivation = Literal["modrelu"]
ComplexVariant = Literal["standard", "widely_linear_phase", "hybrid"]
COMPLEX_ACTIVATIONS: tuple[ComplexActivation, ...] = ("modrelu",)
COMPLEX_VARIANTS: tuple[ComplexVariant, ...] = ("standard", "widely_linear_phase", "hybrid")
COMPLEX_CHANNELS: tuple[int, int, int, int] = (32, 64, 128, 192)
HYBRID_REAL_CHANNELS: tuple[int, int, int, int] = (42, 84, 176, 256)
PARAMETER_MATCHED_REAL_CHANNELS: tuple[int, int, int, int] = (64, 128, 256, 384)


class RealAmplitudeCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        dropout: float = 0.2,
        channels: tuple[int, int, int, int] = PARAMETER_MATCHED_REAL_CHANNELS,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
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
            nn.Dropout(dropout),
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


class WidelyLinearComplexConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.direct_real = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.direct_imag = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conj_real = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.conj_imag = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        with torch.no_grad():
            for conv in (self.direct_real, self.direct_imag, self.conj_real, self.conj_imag):
                conv.weight.mul_(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        real = (
            self.direct_real(x.real)
            - self.direct_imag(x.imag)
            + self.conj_real(x.real)
            + self.conj_imag(x.imag)
        )
        imag = (
            self.direct_imag(x.real)
            + self.direct_real(x.imag)
            + self.conj_imag(x.real)
            - self.conj_real(x.imag)
        )
        return torch.complex(real, imag)


class InputPhaseGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.phase_scale = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        magnitude = torch.abs(x)
        phase = torch.atan2(x.imag, x.real)
        gated_phase = self.phase_scale.view(1, -1, 1, 1) * phase
        return torch.polar(magnitude, gated_phase)


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
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = torch.abs(x)
        bias = self.bias.view(1, -1, 1, 1)
        scale = F.relu(magnitude + bias) / (magnitude + self.eps)
        return x * scale


def build_complex_activation(name: ComplexActivation, channels: int) -> nn.Module:
    if name == "modrelu":
        return ModReLU(channels)
    raise ValueError(f"Unknown complex activation: {name}")


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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activation: ComplexActivation,
        conv_layer: type[nn.Module] = ComplexConv2d,
    ) -> None:
        super().__init__()
        self.conv1 = conv_layer(in_channels, out_channels)
        self.norm1 = ComplexBatchNorm2d(out_channels)
        self.act1 = build_complex_activation(activation, out_channels)
        self.conv2 = conv_layer(out_channels, out_channels)
        self.norm2 = ComplexBatchNorm2d(out_channels)
        self.act2 = build_complex_activation(activation, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class HybridBlock(nn.Module):
    def __init__(
        self,
        real_in_channels: int,
        real_out_channels: int,
        complex_in_channels: int,
        complex_out_channels: int,
        *,
        activation: ComplexActivation,
    ) -> None:
        super().__init__()
        self.real_block = _real_block(real_in_channels, real_out_channels)
        self.complex_block = ComplexBlock(complex_in_channels, complex_out_channels, activation=activation)
        self.complex_to_real = nn.Conv2d(complex_out_channels, real_out_channels, kernel_size=1, bias=False)
        self.real_to_complex = nn.Conv2d(real_out_channels, complex_out_channels, kernel_size=1, bias=False)

    def forward(self, real: torch.Tensor, complex_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real_features = self.real_block(real)
        complex_features = self.complex_block(complex_input)
        real_features = real_features + self.complex_to_real(torch.abs(complex_features))
        complex_skip = self.real_to_complex(real_features)
        complex_features = complex_features + torch.complex(complex_skip, torch.zeros_like(complex_skip))
        return real_features, complex_features


class ComplexT2CNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        dropout: float = 0.2,
        activation: ComplexActivation = "modrelu",
        variant: ComplexVariant = "standard",
    ) -> None:
        super().__init__()
        if variant not in COMPLEX_VARIANTS:
            raise ValueError(f"Unknown complex variant: {variant}")
        c1, c2, c3, c4 = COMPLEX_CHANNELS
        conv_layer = WidelyLinearComplexConv2d if variant == "widely_linear_phase" else ComplexConv2d
        self.input_gate = InputPhaseGate(in_channels) if variant == "widely_linear_phase" else nn.Identity()
        self.block1 = ComplexBlock(in_channels, c1, activation=activation, conv_layer=conv_layer)
        self.pool1 = ComplexAvgPool2d(2)
        self.block2 = ComplexBlock(c1, c2, activation=activation, conv_layer=conv_layer)
        self.pool2 = ComplexAvgPool2d(2)
        self.block3 = ComplexBlock(c2, c3, activation=activation, conv_layer=conv_layer)
        self.pool3 = ComplexAvgPool2d(2)
        self.block4 = ComplexBlock(c3, c4, activation=activation, conv_layer=conv_layer)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_gate(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        magnitude_features = torch.abs(x)
        pooled = F.adaptive_avg_pool2d(magnitude_features, 1).flatten(1)
        return self.classifier(self.dropout(pooled)).squeeze(-1)


class HybridComplexT2CNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        dropout: float = 0.2,
        activation: ComplexActivation = "modrelu",
    ) -> None:
        super().__init__()
        r1, r2, r3, r4 = HYBRID_REAL_CHANNELS
        c1, c2, c3, c4 = COMPLEX_CHANNELS
        self.block1 = HybridBlock(in_channels, r1, in_channels, c1, activation=activation)
        self.pool1 = nn.MaxPool2d(2)
        self.complex_pool1 = ComplexAvgPool2d(2)
        self.block2 = HybridBlock(r1, r2, c1, c2, activation=activation)
        self.pool2 = nn.MaxPool2d(2)
        self.complex_pool2 = ComplexAvgPool2d(2)
        self.block3 = HybridBlock(r2, r3, c2, c3, activation=activation)
        self.pool3 = nn.MaxPool2d(2)
        self.complex_pool3 = ComplexAvgPool2d(2)
        self.block4 = HybridBlock(r3, r4, c3, c4, activation=activation)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(r4 + c4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        real = torch.abs(x)
        complex_features = x
        real, complex_features = self.block1(real, complex_features)
        real = self.pool1(real)
        complex_features = self.complex_pool1(complex_features)
        real, complex_features = self.block2(real, complex_features)
        real = self.pool2(real)
        complex_features = self.complex_pool2(complex_features)
        real, complex_features = self.block3(real, complex_features)
        real = self.pool3(real)
        complex_features = self.complex_pool3(complex_features)
        real, complex_features = self.block4(real, complex_features)
        real_pooled = F.adaptive_avg_pool2d(real, 1).flatten(1)
        complex_pooled = F.adaptive_avg_pool2d(torch.abs(complex_features), 1).flatten(1)
        features = torch.cat((real_pooled, complex_pooled), dim=1)
        return self.classifier(self.dropout(features)).squeeze(-1)


def build_model(
    mode: Literal["real", "complex"],
    *,
    in_channels: int = 5,
    dropout: float = 0.2,
    real_channels: tuple[int, int, int, int] = PARAMETER_MATCHED_REAL_CHANNELS,
    complex_activation: ComplexActivation = "modrelu",
    complex_variant: ComplexVariant = "standard",
) -> nn.Module:
    if mode == "real":
        return RealAmplitudeCNN(in_channels=in_channels, dropout=dropout, channels=real_channels)
    if mode == "complex":
        if complex_variant == "hybrid":
            return HybridComplexT2CNN(in_channels=in_channels, dropout=dropout, activation=complex_activation)
        return ComplexT2CNN(
            in_channels=in_channels,
            dropout=dropout,
            activation=complex_activation,
            variant=complex_variant,
        )
    raise ValueError(f"Unknown model mode: {mode}")
