from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


ComplexPooling = Literal["max", "median", "average"]
ComplexNormalization = Literal["rms", "batchnorm"]
ComplexConvolution = Literal["standard", "widely_linear"]
ComplexStreams = Literal["complex_only", "dual"]
ComplexInteraction = Literal["none", "modulus_gate", "holographic"]
ComplexActivation = Literal["modrelu", "magnitude_silu", "crelu", "cardioid"]


COMPLEX_CHANNELS: tuple[int, int, int, int] = (32, 64, 128, 192)
TARGET_PARAMETER_TOLERANCE = 0.01
# sqrt(2) times the complex widths. A real convolution at these widths has
# approximately the same number of scalar weights as a complex convolution.
PARAMETER_MATCHED_REAL_CHANNELS: tuple[int, int, int, int] = (45, 91, 181, 272)


class RealAmplitudeCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4 = PARAMETER_MATCHED_REAL_CHANNELS
        self.features = nn.Sequential(
            _real_block(4, c1),
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


class WidelyLinearComplexConv2d(nn.Module):
    """Widely-linear convolution ``W1 * z + W2 * conj(z)`` without bias."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.direct_real = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.direct_imag = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.conjugate_real = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        self.conjugate_imag = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False
        )
        with torch.no_grad():
            for convolution in (
                self.direct_real,
                self.direct_imag,
                self.conjugate_real,
                self.conjugate_imag,
            ):
                convolution.weight.mul_(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        real = (
            self.direct_real(x.real)
            - self.direct_imag(x.imag)
            + self.conjugate_real(x.real)
            + self.conjugate_imag(x.imag)
        )
        imag = (
            self.direct_imag(x.real)
            + self.direct_real(x.imag)
            + self.conjugate_imag(x.real)
            - self.conjugate_real(x.imag)
        )
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


class ComplexBatchNorm2d(nn.Module):
    """Trabelsi-style per-channel whitening of real and imaginary components."""

    def __init__(self, channels: int, *, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        initial_variance = 1 / math.sqrt(2)
        self.gamma_rr = nn.Parameter(torch.full((channels,), initial_variance))
        self.gamma_ii = nn.Parameter(torch.full((channels,), initial_variance))
        self.gamma_ri = nn.Parameter(torch.zeros(channels))
        self.beta_real = nn.Parameter(torch.zeros(channels))
        self.beta_imag = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean_real", torch.zeros(channels))
        self.register_buffer("running_mean_imag", torch.zeros(channels))
        self.register_buffer("running_covar_rr", torch.full((channels,), initial_variance))
        self.register_buffer("running_covar_ii", torch.full((channels,), initial_variance))
        self.register_buffer("running_covar_ri", torch.zeros(channels))
        self.eps = eps
        self.momentum = momentum

    @staticmethod
    def _view(values: torch.Tensor) -> torch.Tensor:
        return values.view(1, -1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x) or x.ndim != 4:
            raise TypeError("ComplexBatchNorm2d expects a rank-four complex tensor")
        reduction_dims = (0, 2, 3)
        if self.training:
            mean_real = x.real.mean(dim=reduction_dims)
            mean_imag = x.imag.mean(dim=reduction_dims)
            centered_real = x.real - self._view(mean_real)
            centered_imag = x.imag - self._view(mean_imag)
            covar_rr = centered_real.square().mean(dim=reduction_dims)
            covar_ii = centered_imag.square().mean(dim=reduction_dims)
            covar_ri = (centered_real * centered_imag).mean(dim=reduction_dims)
            with torch.no_grad():
                self.running_mean_real.lerp_(mean_real.detach(), self.momentum)
                self.running_mean_imag.lerp_(mean_imag.detach(), self.momentum)
                self.running_covar_rr.lerp_(covar_rr.detach(), self.momentum)
                self.running_covar_ii.lerp_(covar_ii.detach(), self.momentum)
                self.running_covar_ri.lerp_(covar_ri.detach(), self.momentum)
        else:
            centered_real = x.real - self._view(self.running_mean_real)
            centered_imag = x.imag - self._view(self.running_mean_imag)
            covar_rr = self.running_covar_rr
            covar_ii = self.running_covar_ii
            covar_ri = self.running_covar_ri

        variance_rr = covar_rr + self.eps
        variance_ii = covar_ii + self.eps
        determinant = (variance_rr * variance_ii - covar_ri.square()).clamp_min(self.eps**2)
        root_determinant = torch.sqrt(determinant)
        root_trace = torch.sqrt(variance_rr + variance_ii + 2 * root_determinant)
        inverse_scale = torch.reciprocal(root_determinant * root_trace)
        whiten_rr = (variance_ii + root_determinant) * inverse_scale
        whiten_ii = (variance_rr + root_determinant) * inverse_scale
        whiten_ri = -covar_ri * inverse_scale
        normalized_real = self._view(whiten_rr) * centered_real + self._view(whiten_ri) * centered_imag
        normalized_imag = self._view(whiten_ri) * centered_real + self._view(whiten_ii) * centered_imag
        output_real = (
            self._view(self.gamma_rr) * normalized_real
            + self._view(self.gamma_ri) * normalized_imag
            + self._view(self.beta_real)
        )
        output_imag = (
            self._view(self.gamma_ri) * normalized_real
            + self._view(self.gamma_ii) * normalized_imag
            + self._view(self.beta_imag)
        )
        return torch.complex(output_real, output_imag)


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


class MagnitudeGatedSiLU(nn.Module):
    """Smooth phase-equivariant gate ``z * sigmoid(a * (|z| - b))``."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.log_slope = nn.Parameter(torch.full((channels,), 0.5413248546))
        self.threshold = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        slope = F.softplus(self.log_slope).view(1, -1, 1, 1)
        threshold = self.threshold.view(1, -1, 1, 1)
        return x * torch.sigmoid(slope * (torch.abs(x) - threshold))


class ComplexReLU(nn.Module):
    """Apply ReLU independently to the real and imaginary components."""

    def __init__(self, channels: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(F.relu(x.real), F.relu(x.imag))


class CardioidActivation(nn.Module):
    """Phase-selective cardioid activation ``0.5 * (1 + cos(angle(z))) * z``."""

    def __init__(self, channels: int) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 + torch.cos(torch.angle(x))) * x


def build_complex_activation(activation: ComplexActivation, channels: int) -> nn.Module:
    if activation == "modrelu":
        return ModReLU(channels)
    if activation == "magnitude_silu":
        return MagnitudeGatedSiLU(channels)
    if activation == "crelu":
        return ComplexReLU(channels)
    if activation == "cardioid":
        return CardioidActivation(channels)
    raise ValueError(f"Unknown complex activation: {activation}")


def _complex_norm(channels: int, normalization: ComplexNormalization) -> nn.Module:
    if normalization == "rms":
        return ComplexRMSNorm2d(channels)
    if normalization == "batchnorm":
        return ComplexBatchNorm2d(channels)
    raise ValueError(f"Unknown complex normalization: {normalization}")


def _complex_convolution(convolution: ComplexConvolution) -> type[nn.Module]:
    if convolution == "standard":
        return ComplexConv2d
    if convolution == "widely_linear":
        return WidelyLinearComplexConv2d
    raise ValueError(f"Unknown complex convolution: {convolution}")


class InterferenceAwareHolographicAttention2d(nn.Module):
    """Complex self-attention with Hermitian similarity and amplitude penalty."""

    def __init__(
        self,
        channels: int,
        attention_channels: int,
        *,
        normalization: ComplexNormalization,
        convolution: ComplexConvolution,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        if attention_channels < 1:
            raise ValueError("attention_channels must be positive")
        conv = _complex_convolution(convolution)
        self.attention_channels = attention_channels
        self.query = conv(channels, attention_channels, kernel_size=1, padding=0)
        self.key = conv(channels, attention_channels, kernel_size=1, padding=0)
        self.value = conv(channels, attention_channels, kernel_size=1, padding=0)
        self.output = conv(attention_channels, channels, kernel_size=1, padding=0)
        self.norm = _complex_norm(channels, normalization)
        self.register_buffer("gamma", torch.tensor(float(gamma)))

    def interference_logits(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(query) or not torch.is_complex(key):
            raise TypeError("Holographic queries and keys must be complex")
        if query.shape != key.shape or query.ndim != 3:
            raise ValueError("Queries and keys must share shape (batch, tokens, channels)")
        scale = math.sqrt(self.attention_channels)
        hermitian_similarity = torch.matmul(query, key.conj().transpose(-2, -1)).real / scale
        query_magnitude = torch.abs(query)
        key_magnitude = torch.abs(key)
        magnitude_distance = (
            query_magnitude.square().sum(dim=-1, keepdim=True)
            + key_magnitude.square().sum(dim=-1).unsqueeze(-2)
            - 2 * torch.matmul(query_magnitude, key_magnitude.transpose(-2, -1))
        ).clamp_min(0) / self.attention_channels
        return hermitian_similarity - self.gamma * magnitude_distance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x) or x.ndim != 4:
            raise TypeError("Holographic attention expects a rank-four complex tensor")
        batch, _, height, width = x.shape
        query = self.query(x).flatten(2).transpose(1, 2)
        key = self.key(x).flatten(2).transpose(1, 2)
        value = self.value(x).flatten(2).transpose(1, 2)
        attention = torch.softmax(self.interference_logits(query, key), dim=-1)
        aggregated = torch.complex(
            torch.matmul(attention, value.real),
            torch.matmul(attention, value.imag),
        ).transpose(1, 2).reshape(batch, self.attention_channels, height, width)
        return self.norm(x + self.output(aggregated))


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


class ComplexMagnitudeMedianPool2d(nn.Module):
    """Select the full complex value with median-ranked amplitude per window.

    For even-sized windows, ``torch.median`` selects the lower of the two
    central amplitudes. Selecting an observed value, rather than averaging the
    two central values, retains that activation's original complex phase.
    """

    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("ComplexMagnitudeMedianPool2d expects a complex tensor")

        real_patches = F.unfold(
            x.real,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        imag_patches = F.unfold(
            x.imag,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        batch, channels, height, width = x.shape
        window_elements = self.kernel_size**2
        locations = real_patches.shape[-1]
        real_patches = real_patches.reshape(
            batch, channels, window_elements, locations
        ).transpose(2, 3)
        imag_patches = imag_patches.reshape(
            batch, channels, window_elements, locations
        ).transpose(2, 3)
        amplitude_squared = real_patches.square() + imag_patches.square()
        median_indices = amplitude_squared.median(dim=-1).indices.unsqueeze(-1)
        pooled_real = torch.gather(real_patches, -1, median_indices).squeeze(-1)
        pooled_imag = torch.gather(imag_patches, -1, median_indices).squeeze(-1)
        output_height = (height - self.kernel_size) // self.stride + 1
        output_width = (width - self.kernel_size) // self.stride + 1
        output_shape = (batch, channels, output_height, output_width)
        return torch.complex(
            pooled_real.reshape(output_shape),
            pooled_imag.reshape(output_shape),
        )


class ComplexAveragePool2d(nn.Module):
    """Average real and imaginary components over each pooling window."""

    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("ComplexAveragePool2d expects a complex tensor")
        return torch.complex(
            F.avg_pool2d(x.real, self.kernel_size, stride=self.stride),
            F.avg_pool2d(x.imag, self.kernel_size, stride=self.stride),
        )


def build_complex_pool(pooling: ComplexPooling) -> nn.Module:
    if pooling == "max":
        return ComplexMagnitudeMaxPool2d(2)
    if pooling == "median":
        return ComplexMagnitudeMedianPool2d(2)
    if pooling == "average":
        return ComplexAveragePool2d(2)
    raise ValueError(f"Unknown complex pooling mode: {pooling}")


class ComplexBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        normalization: ComplexNormalization,
        convolution: ComplexConvolution,
        activation: ComplexActivation,
    ) -> None:
        super().__init__()
        conv = _complex_convolution(convolution)
        self.conv1 = conv(in_channels, out_channels)
        self.norm1 = _complex_norm(out_channels, normalization)
        self.act1 = build_complex_activation(activation, out_channels)
        self.conv2 = conv(out_channels, out_channels)
        self.norm2 = _complex_norm(out_channels, normalization)
        self.act2 = build_complex_activation(activation, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        return self.act2(self.norm2(self.conv2(x)))


class ComplexT2CNN(nn.Module):
    def __init__(
        self,
        *,
        pooling: ComplexPooling = "max",
        normalization: ComplexNormalization = "rms",
        convolution: ComplexConvolution = "standard",
        activation: ComplexActivation = "modrelu",
        interaction: ComplexInteraction = "none",
        channels: tuple[int, int, int, int] = COMPLEX_CHANNELS,
    ) -> None:
        super().__init__()
        if interaction not in ("none", "holographic"):
            raise ValueError(f"Complex-only streams do not support interaction={interaction}")
        self.model_channels = channels
        self.block1 = ComplexBlock(
            4, channels[0], normalization=normalization, convolution=convolution, activation=activation
        )
        self.pool1 = build_complex_pool(pooling)
        self.block2 = ComplexBlock(
            channels[0], channels[1], normalization=normalization, convolution=convolution, activation=activation
        )
        self.pool2 = build_complex_pool(pooling)
        self.block3 = ComplexBlock(
            channels[1], channels[2], normalization=normalization, convolution=convolution, activation=activation
        )
        self.pool3 = build_complex_pool(pooling)
        self.holographic_attention = (
            InterferenceAwareHolographicAttention2d(
                channels[2],
                max(4, channels[2] // 4),
                normalization=normalization,
                convolution=convolution,
            )
            if interaction == "holographic"
            else nn.Identity()
        )
        self.block4 = ComplexBlock(
            channels[2], channels[3], normalization=normalization, convolution=convolution, activation=activation
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(channels[3], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.holographic_attention(x)
        x = self.block4(x)
        pooled = F.adaptive_avg_pool2d(torch.abs(x), 1).flatten(1)
        return self.classifier(self.dropout(pooled)).squeeze(-1)


class ModulusCrossStreamGate(nn.Module):
    def __init__(self, complex_channels: int, real_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(complex_channels, real_channels, kernel_size=3, padding=1)

    def forward(self, complex_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.convolution(torch.abs(complex_features)))


class DualStreamBlock(nn.Module):
    def __init__(
        self,
        real_in_channels: int,
        real_out_channels: int,
        complex_in_channels: int,
        complex_out_channels: int,
        *,
        normalization: ComplexNormalization,
        convolution: ComplexConvolution,
        activation: ComplexActivation,
        gated: bool,
    ) -> None:
        super().__init__()
        self.real_block = _real_block(real_in_channels, real_out_channels)
        self.complex_block = ComplexBlock(
            complex_in_channels,
            complex_out_channels,
            normalization=normalization,
            convolution=convolution,
            activation=activation,
        )
        self.gate = (
            ModulusCrossStreamGate(complex_out_channels, real_out_channels)
            if gated
            else None
        )

    def forward(
        self, real_features: torch.Tensor, complex_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_features = self.real_block(real_features)
        complex_features = self.complex_block(complex_features)
        if self.gate is not None:
            real_features = real_features * self.gate(complex_features)
        return real_features, complex_features


class DualStreamComplexT2CNN(nn.Module):
    def __init__(
        self,
        *,
        pooling: ComplexPooling,
        normalization: ComplexNormalization,
        convolution: ComplexConvolution,
        activation: ComplexActivation,
        interaction: ComplexInteraction,
        channels: tuple[int, int, int, int],
    ) -> None:
        super().__init__()
        if interaction not in ("none", "modulus_gate", "holographic"):
            raise ValueError(f"Unknown dual-stream interaction: {interaction}")
        self.model_channels = channels
        gated = interaction == "modulus_gate"
        self.block1 = DualStreamBlock(
            4, channels[0], 4, channels[0], normalization=normalization,
            convolution=convolution, activation=activation, gated=gated,
        )
        self.block2 = DualStreamBlock(
            channels[0], channels[1], channels[0], channels[1], normalization=normalization,
            convolution=convolution, activation=activation, gated=gated,
        )
        self.block3 = DualStreamBlock(
            channels[1], channels[2], channels[1], channels[2], normalization=normalization,
            convolution=convolution, activation=activation, gated=gated,
        )
        self.block4 = DualStreamBlock(
            channels[2], channels[3], channels[2], channels[3], normalization=normalization,
            convolution=convolution, activation=activation, gated=gated,
        )
        self.real_pool1 = nn.MaxPool2d(2)
        self.real_pool2 = nn.MaxPool2d(2)
        self.real_pool3 = nn.MaxPool2d(2)
        self.complex_pool1 = build_complex_pool(pooling)
        self.complex_pool2 = build_complex_pool(pooling)
        self.complex_pool3 = build_complex_pool(pooling)
        self.holographic_attention = (
            InterferenceAwareHolographicAttention2d(
                channels[2], max(4, channels[2] // 4),
                normalization=normalization, convolution=convolution,
            )
            if interaction == "holographic"
            else nn.Identity()
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(2 * channels[3], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        real_features, complex_features = torch.abs(x), x
        real_features, complex_features = self.block1(real_features, complex_features)
        real_features = self.real_pool1(real_features)
        complex_features = self.complex_pool1(complex_features)
        real_features, complex_features = self.block2(real_features, complex_features)
        real_features = self.real_pool2(real_features)
        complex_features = self.complex_pool2(complex_features)
        real_features, complex_features = self.block3(real_features, complex_features)
        real_features = self.real_pool3(real_features)
        complex_features = self.complex_pool3(complex_features)
        complex_features = self.holographic_attention(complex_features)
        real_features, complex_features = self.block4(real_features, complex_features)
        pooled = torch.cat(
            (
                F.adaptive_avg_pool2d(real_features, 1).flatten(1),
                F.adaptive_avg_pool2d(torch.abs(complex_features), 1).flatten(1),
            ),
            dim=1,
        )
        return self.classifier(self.dropout(pooled)).squeeze(-1)


def _trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _make_complex_model(
    *,
    pooling: ComplexPooling,
    normalization: ComplexNormalization,
    convolution: ComplexConvolution,
    streams: ComplexStreams,
    interaction: ComplexInteraction,
    activation: ComplexActivation,
    channels: tuple[int, int, int, int],
) -> nn.Module:
    if streams == "complex_only":
        return ComplexT2CNN(
            pooling=pooling,
            normalization=normalization,
            convolution=convolution,
            activation=activation,
            interaction=interaction,
            channels=channels,
        )
    if streams == "dual":
        return DualStreamComplexT2CNN(
            pooling=pooling,
            normalization=normalization,
            convolution=convolution,
            activation=activation,
            interaction=interaction,
            channels=channels,
        )
    raise ValueError(f"Unknown complex stream layout: {streams}")


def _scaled_channels(scale: float) -> tuple[int, int, int, int]:
    return tuple(max(4, round(channel * scale)) for channel in COMPLEX_CHANNELS)  # type: ignore[return-value]


def matched_complex_channels(
    *,
    normalization: ComplexNormalization,
    convolution: ComplexConvolution,
    streams: ComplexStreams,
    interaction: ComplexInteraction,
    activation: ComplexActivation,
) -> tuple[int, int, int, int]:
    """Select dependent widths that most closely match the fixed real baseline."""
    if (
        normalization == "rms"
        and convolution == "standard"
        and streams == "complex_only"
        and interaction == "none"
        and activation == "modrelu"
    ):
        return COMPLEX_CHANNELS

    with torch.random.fork_rng(devices=[]):
        target = _trainable_parameters(RealAmplitudeCNN())

        def count(scale: float) -> tuple[int, tuple[int, int, int, int]]:
            channels = _scaled_channels(scale)
            model = _make_complex_model(
                pooling="average",
                normalization=normalization,
                convolution=convolution,
                streams=streams,
                interaction=interaction,
                activation=activation,
                channels=channels,
            )
            return _trainable_parameters(model), channels

        low, high = 0.25, 1.25
        for _ in range(14):
            middle = (low + high) / 2
            parameters, _ = count(middle)
            if parameters < target:
                low = middle
            else:
                high = middle
        center = (low + high) / 2
        candidates = {
            _scaled_channels(center + offset / 1000)
            for offset in range(-20, 21)
            if center + offset / 1000 > 0
        }
        def candidate_key(channels: tuple[int, int, int, int]):
            parameters = _trainable_parameters(
                _make_complex_model(
                    pooling="average",
                    normalization=normalization,
                    convolution=convolution,
                    streams=streams,
                    interaction=interaction,
                    activation=activation,
                    channels=channels,
                )
            )
            return abs(parameters - target), channels

        best_channels = min(candidates, key=candidate_key)
        best_parameters = _trainable_parameters(
            _make_complex_model(
                pooling="average",
                normalization=normalization,
                convolution=convolution,
                streams=streams,
                interaction=interaction,
                activation=activation,
                channels=best_channels,
            )
        )
        if abs(best_parameters - target) / target > TARGET_PARAMETER_TOLERANCE:
            raise RuntimeError(
                f"Could not parameter-match complex model; best channels={best_channels}, "
                f"parameters={best_parameters}, target={target}."
            )
    return best_channels


def build_model(
    mode: Literal["real", "complex"],
    *,
    complex_pooling: ComplexPooling = "max",
    complex_normalization: ComplexNormalization = "rms",
    complex_convolution: ComplexConvolution = "standard",
    complex_streams: ComplexStreams = "complex_only",
    complex_interaction: ComplexInteraction = "none",
    complex_activation: ComplexActivation = "modrelu",
) -> nn.Module:
    if mode == "real":
        return RealAmplitudeCNN()
    if mode != "complex":
        raise ValueError(f"Unknown model mode: {mode}")
    if complex_streams == "complex_only" and complex_interaction == "modulus_gate":
        raise ValueError("modulus_gate requires dual streams")
    channels = matched_complex_channels(
        normalization=complex_normalization,
        convolution=complex_convolution,
        streams=complex_streams,
        interaction=complex_interaction,
        activation=complex_activation,
    )
    return _make_complex_model(
        pooling=complex_pooling,
        normalization=complex_normalization,
        convolution=complex_convolution,
        streams=complex_streams,
        interaction=complex_interaction,
        activation=complex_activation,
        channels=channels,
    )
