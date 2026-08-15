from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


ComplexPooling = Literal["max", "median", "average"]
ComplexNormalization = Literal["rms", "batchnorm"]


COMPLEX_CHANNELS: tuple[int, int, int, int] = (32, 64, 128, 192)
# Widely-linear convolutions have twice as many scalar kernels as ordinary
# complex convolutions, so 1/sqrt(2)-scaled widths preserve the model budget.
WIDELY_LINEAR_CHANNELS: tuple[int, int, int, int] = (23, 45, 91, 136)
# Two reduced streams plus each 3x3 cross-stream gate approximately match the
# scalar parameter count of the single-stream real and complex baselines.
MODULUS_GATED_CHANNELS: tuple[int, int, int, int] = (24, 47, 95, 143)
# The slightly narrower final block offsets the Q/K/V/output projections.
HOLOGRAPHIC_CHANNELS: tuple[int, int, int, int] = (32, 64, 128, 188)
HOLOGRAPHIC_ATTENTION_CHANNELS = 32
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
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.direct_imag = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.conjugate_real = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.conjugate_imag = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
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
    """Trabelsi-style complex batch normalization with 2x2 whitening.

    Each complex channel is represented by its real and imaginary components.
    The centered components are whitened by the inverse square root of their
    2x2 covariance matrix, then transformed by a learned symmetric matrix and
    complex shift.
    """

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
        if not torch.is_complex(x):
            raise TypeError("ComplexBatchNorm2d expects a complex tensor")
        if x.ndim != 4:
            raise ValueError("ComplexBatchNorm2d expects a tensor with shape (batch, channels, height, width)")

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
            mean_real = self.running_mean_real
            mean_imag = self.running_mean_imag
            covar_rr = self.running_covar_rr
            covar_ii = self.running_covar_ii
            covar_ri = self.running_covar_ri
            centered_real = x.real - self._view(mean_real)
            centered_imag = x.imag - self._view(mean_imag)

        # Principal inverse square root of the regularized symmetric 2x2
        # covariance matrix. This closed form avoids an eigendecomposition and
        # remains differentiable when the two eigenvalues are nearly equal.
        variance_rr = covar_rr + self.eps
        variance_ii = covar_ii + self.eps
        determinant = (variance_rr * variance_ii - covar_ri.square()).clamp_min(
            self.eps**2
        )
        root_determinant = torch.sqrt(determinant)
        root_trace = torch.sqrt(variance_rr + variance_ii + 2 * root_determinant)
        inverse_scale = torch.reciprocal(root_determinant * root_trace)
        whiten_rr = (variance_ii + root_determinant) * inverse_scale
        whiten_ii = (variance_rr + root_determinant) * inverse_scale
        whiten_ri = -covar_ri * inverse_scale

        normalized_real = (
            self._view(whiten_rr) * centered_real
            + self._view(whiten_ri) * centered_imag
        )
        normalized_imag = (
            self._view(whiten_ri) * centered_real
            + self._view(whiten_ii) * centered_imag
        )

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


class InterferenceAwareHolographicAttention2d(nn.Module):
    """Complex self-attention with Hermitian similarity and amplitude penalty."""

    def __init__(
        self,
        channels: int,
        attention_channels: int,
        *,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        if attention_channels < 1:
            raise ValueError("attention_channels must be positive")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.attention_channels = attention_channels
        self.query = ComplexConv2d(
            channels,
            attention_channels,
            kernel_size=1,
            padding=0,
        )
        self.key = ComplexConv2d(
            channels,
            attention_channels,
            kernel_size=1,
            padding=0,
        )
        self.value = ComplexConv2d(
            channels,
            attention_channels,
            kernel_size=1,
            padding=0,
        )
        self.output = ComplexConv2d(
            attention_channels,
            channels,
            kernel_size=1,
            padding=0,
        )
        self.norm = ComplexRMSNorm2d(channels)
        self.register_buffer("gamma", torch.tensor(float(gamma)))

    def interference_logits(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        """Return corrected interference logits for ``(batch, tokens, channels)``."""
        if not torch.is_complex(query) or not torch.is_complex(key):
            raise TypeError("Holographic queries and keys must be complex")
        if query.shape != key.shape or query.ndim != 3:
            raise ValueError("Queries and keys must share shape (batch, tokens, channels)")
        if query.shape[-1] != self.attention_channels:
            raise ValueError("Query/key channel count does not match attention_channels")

        scale = math.sqrt(self.attention_channels)
        hermitian_similarity = torch.matmul(
            query,
            key.conj().transpose(-2, -1),
        ).real / scale

        query_magnitude = torch.abs(query)
        key_magnitude = torch.abs(key)
        query_power = query_magnitude.square().sum(dim=-1, keepdim=True)
        key_power = key_magnitude.square().sum(dim=-1).unsqueeze(-2)
        magnitude_cross = torch.matmul(
            query_magnitude,
            key_magnitude.transpose(-2, -1),
        )
        magnitude_distance = (
            query_power + key_power - 2 * magnitude_cross
        ).clamp_min(0) / self.attention_channels
        return hermitian_similarity - self.gamma * magnitude_distance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("InterferenceAwareHolographicAttention2d expects a complex tensor")
        if x.ndim != 4:
            raise ValueError("Holographic attention expects shape (batch, channels, height, width)")

        batch, _, height, width = x.shape
        query = self.query(x).flatten(2).transpose(1, 2)
        key = self.key(x).flatten(2).transpose(1, 2)
        value = self.value(x).flatten(2).transpose(1, 2)
        attention = torch.softmax(self.interference_logits(query, key), dim=-1)
        aggregated = torch.complex(
            torch.matmul(attention, value.real),
            torch.matmul(attention, value.imag),
        )
        aggregated = aggregated.transpose(1, 2).reshape(
            batch,
            self.attention_channels,
            height,
            width,
        )
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
        normalization: ComplexNormalization = "rms",
        convolution: type[nn.Module] = ComplexConv2d,
    ) -> None:
        super().__init__()
        self.conv1 = convolution(in_channels, out_channels)
        self.norm1 = _complex_norm(out_channels, normalization)
        self.act1 = ModReLU(out_channels)
        self.conv2 = convolution(out_channels, out_channels)
        self.norm2 = _complex_norm(out_channels, normalization)
        self.act2 = ModReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


def _complex_norm(channels: int, normalization: ComplexNormalization) -> nn.Module:
    if normalization == "rms":
        return ComplexRMSNorm2d(channels)
    if normalization == "batchnorm":
        return ComplexBatchNorm2d(channels)
    raise ValueError(f"Unknown complex normalization: {normalization}")


class ComplexT2CNN(nn.Module):
    def __init__(
        self,
        *,
        pooling: ComplexPooling = "max",
        normalization: ComplexNormalization = "rms",
        channels: tuple[int, int, int, int] = COMPLEX_CHANNELS,
        convolution: type[nn.Module] = ComplexConv2d,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.block1 = ComplexBlock(
            4,
            c1,
            normalization=normalization,
            convolution=convolution,
        )
        self.pool1 = build_complex_pool(pooling)
        self.block2 = ComplexBlock(
            c1,
            c2,
            normalization=normalization,
            convolution=convolution,
        )
        self.pool2 = build_complex_pool(pooling)
        self.block3 = ComplexBlock(
            c2,
            c3,
            normalization=normalization,
            convolution=convolution,
        )
        self.pool3 = build_complex_pool(pooling)
        self.block4 = ComplexBlock(
            c3,
            c4,
            normalization=normalization,
            convolution=convolution,
        )
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(c4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.block4(x)
        pooled = F.adaptive_avg_pool2d(torch.abs(x), 1).flatten(1)
        return self.classifier(self.dropout(pooled)).squeeze(-1)


class ComplexKSpaceCNN(ComplexT2CNN):
    """Complex classifier operating directly on k-space with average pooling."""

    def __init__(self) -> None:
        super().__init__(pooling="average")


class ComplexBatchNormKSpaceCNN(ComplexT2CNN):
    """K-space classifier using Trabelsi-style complex batch normalization."""

    def __init__(self) -> None:
        super().__init__(pooling="average", normalization="batchnorm")


class WidelyLinearComplexT2CNN(ComplexT2CNN):
    """RMS-normalized widely-linear model with complex average pooling."""

    def __init__(self) -> None:
        super().__init__(
            pooling="average",
            normalization="rms",
            channels=WIDELY_LINEAR_CHANNELS,
            convolution=WidelyLinearComplexConv2d,
        )


class HolographicAttentionT2CNN(ComplexT2CNN):
    """RMS-normalized complex CNN with corrected holographic self-attention."""

    def __init__(self) -> None:
        super().__init__(
            pooling="average",
            normalization="rms",
            channels=HOLOGRAPHIC_CHANNELS,
        )
        self.holographic_attention = InterferenceAwareHolographicAttention2d(
            HOLOGRAPHIC_CHANNELS[2],
            HOLOGRAPHIC_ATTENTION_CHANNELS,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.holographic_attention(x)
        x = self.block4(x)
        pooled = F.adaptive_avg_pool2d(torch.abs(x), 1).flatten(1)
        return self.classifier(self.dropout(pooled)).squeeze(-1)


class ModulusCrossStreamGate(nn.Module):
    """Generate a real-valued gate from the modulus of a complex feature map."""

    def __init__(self, complex_channels: int, real_channels: int) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(
            complex_channels,
            real_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, complex_features: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(complex_features):
            raise TypeError("ModulusCrossStreamGate expects a complex tensor")
        return torch.sigmoid(self.convolution(torch.abs(complex_features)))


class ModulusGatedBlock(nn.Module):
    """Update real and complex streams, then gate the real stream by |z|."""

    def __init__(
        self,
        real_in_channels: int,
        real_out_channels: int,
        complex_in_channels: int,
        complex_out_channels: int,
    ) -> None:
        super().__init__()
        self.real_block = _real_block(real_in_channels, real_out_channels)
        self.complex_block = ComplexBlock(
            complex_in_channels,
            complex_out_channels,
            normalization="rms",
        )
        self.complex_to_real_gate = ModulusCrossStreamGate(
            complex_out_channels,
            real_out_channels,
        )

    def forward(
        self,
        real_features: torch.Tensor,
        complex_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_features = self.real_block(real_features)
        complex_features = self.complex_block(complex_features)
        gate = self.complex_to_real_gate(complex_features)
        return real_features * gate, complex_features


class ModulusGatedT2CNN(nn.Module):
    """Dual-stream model with complex-modulus gates on the real stream."""

    def __init__(self) -> None:
        super().__init__()
        r1, r2, r3, r4 = MODULUS_GATED_CHANNELS
        c1, c2, c3, c4 = MODULUS_GATED_CHANNELS
        self.block1 = ModulusGatedBlock(4, r1, 4, c1)
        self.real_pool1 = nn.MaxPool2d(2)
        self.complex_pool1 = ComplexAveragePool2d(2)
        self.block2 = ModulusGatedBlock(r1, r2, c1, c2)
        self.real_pool2 = nn.MaxPool2d(2)
        self.complex_pool2 = ComplexAveragePool2d(2)
        self.block3 = ModulusGatedBlock(r2, r3, c2, c3)
        self.real_pool3 = nn.MaxPool2d(2)
        self.complex_pool3 = ComplexAveragePool2d(2)
        self.block4 = ModulusGatedBlock(r3, r4, c3, c4)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(r4 + c4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        real_features = torch.abs(x)
        complex_features = x
        real_features, complex_features = self.block1(
            real_features,
            complex_features,
        )
        real_features = self.real_pool1(real_features)
        complex_features = self.complex_pool1(complex_features)
        real_features, complex_features = self.block2(
            real_features,
            complex_features,
        )
        real_features = self.real_pool2(real_features)
        complex_features = self.complex_pool2(complex_features)
        real_features, complex_features = self.block3(
            real_features,
            complex_features,
        )
        real_features = self.real_pool3(real_features)
        complex_features = self.complex_pool3(complex_features)
        real_features, complex_features = self.block4(
            real_features,
            complex_features,
        )
        real_pooled = F.adaptive_avg_pool2d(real_features, 1).flatten(1)
        complex_pooled = F.adaptive_avg_pool2d(
            torch.abs(complex_features),
            1,
        ).flatten(1)
        pooled = torch.cat((real_pooled, complex_pooled), dim=1)
        return self.classifier(self.dropout(pooled)).squeeze(-1)


def build_model(
    mode: Literal[
        "real",
        "complex",
        "complex_kspace",
        "complex_kspace_batchnorm",
        "complex_widely_linear",
        "complex_modulus_gated",
        "complex_holographic_attention",
    ],
    *,
    complex_pooling: ComplexPooling = "max",
) -> nn.Module:
    if mode == "real":
        return RealAmplitudeCNN()
    if mode == "complex":
        return ComplexT2CNN(pooling=complex_pooling)
    if mode == "complex_kspace":
        return ComplexKSpaceCNN()
    if mode == "complex_kspace_batchnorm":
        return ComplexBatchNormKSpaceCNN()
    if mode == "complex_widely_linear":
        return WidelyLinearComplexT2CNN()
    if mode == "complex_modulus_gated":
        return ModulusGatedT2CNN()
    if mode == "complex_holographic_attention":
        return HolographicAttentionT2CNN()
    raise ValueError(f"Unknown model mode: {mode}")
