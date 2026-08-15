import torch
import pytest

from prost_t2_classification.models import (
    COMPLEX_CHANNELS,
    ComplexAveragePool2d,
    ComplexBatchNorm2d,
    ComplexBatchNormKSpaceCNN,
    ComplexKSpaceCNN,
    ComplexMagnitudeMaxPool2d,
    ComplexMagnitudeMedianPool2d,
    ComplexRMSNorm2d,
    HolographicAttentionT2CNN,
    InterferenceAwareHolographicAttention2d,
    ModulusCrossStreamGate,
    ModulusGatedT2CNN,
    ModReLU,
    PARAMETER_MATCHED_REAL_CHANNELS,
    WidelyLinearComplexConv2d,
    WidelyLinearComplexT2CNN,
    build_model,
)


def _trainable_params(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def test_complex_model_builds():
    x = torch.complex(torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32))

    model = build_model("complex")
    model.eval()
    with torch.no_grad():
        output = model(x)

    assert output.shape == (2,)


def test_modrelu_starts_without_negative_gate_bias():
    activation = ModReLU(4)

    assert torch.all(activation.bias == 0)


def test_widely_linear_convolution_matches_w1_z_plus_w2_conjugate_z():
    convolution = WidelyLinearComplexConv2d(
        1,
        1,
        kernel_size=1,
        padding=0,
    )
    with torch.no_grad():
        convolution.direct_real.weight.fill_(2.0)
        convolution.direct_imag.weight.fill_(3.0)
        convolution.conjugate_real.weight.fill_(5.0)
        convolution.conjugate_imag.weight.fill_(7.0)
    x = torch.tensor([[[[11.0 + 13.0j]]]], dtype=torch.complex64)

    output = convolution(x)
    expected = (2.0 + 3.0j) * x + (5.0 + 7.0j) * x.conj()

    assert torch.allclose(output, expected)
    assert all(
        layer.bias is None
        for layer in (
            convolution.direct_real,
            convolution.direct_imag,
            convolution.conjugate_real,
            convolution.conjugate_imag,
        )
    )


def test_modulus_cross_stream_gate_uses_bounded_real_modulus_gate():
    gate = ModulusCrossStreamGate(2, 3)
    with torch.no_grad():
        gate.convolution.weight.zero_()
        gate.convolution.bias.zero_()
    x = torch.complex(torch.randn(1, 2, 8, 8), torch.randn(1, 2, 8, 8))

    output = gate(x)

    assert output.dtype == torch.float32
    assert output.shape == (1, 3, 8, 8)
    assert torch.allclose(output, torch.full_like(output, 0.5))


def test_holographic_logits_keep_constructive_and_destructive_phase_distinct():
    attention = InterferenceAwareHolographicAttention2d(1, 1, gamma=1.0)
    query = torch.tensor([[[1.0 + 0.0j], [1.0 + 0.0j]]])
    key = torch.tensor([[[1.0 + 0.0j], [-1.0 + 0.0j]]])

    logits = attention.interference_logits(query, key)

    expected = torch.tensor([[[1.0, -1.0], [1.0, -1.0]]])
    assert torch.allclose(logits, expected)


def test_holographic_logits_penalize_amplitude_mismatch_additively():
    attention = InterferenceAwareHolographicAttention2d(1, 1, gamma=2.0)
    query = torch.tensor([[[1.0 + 0.0j], [1.0 + 0.0j]]])
    key = torch.tensor([[[1.0 + 0.0j], [2.0 + 0.0j]]])

    logits = attention.interference_logits(query, key)

    expected = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    assert torch.allclose(logits, expected)


def test_holographic_attention_is_global_phase_equivariant():
    attention = InterferenceAwareHolographicAttention2d(4, 3)
    attention.eval()
    x = torch.complex(torch.randn(2, 4, 8, 8), torch.randn(2, 4, 8, 8))
    rotation = torch.polar(torch.tensor(1.0), torch.tensor(0.731))

    with torch.no_grad():
        output = attention(x)
        rotated_output = attention(x * rotation)

    assert torch.allclose(rotated_output, output * rotation, atol=1e-5, rtol=1e-5)


def test_complex_batch_norm_zero_centers_and_whitens_correlated_components():
    generator = torch.Generator().manual_seed(73191)
    first = torch.randn(4, 3, 16, 16, generator=generator)
    second = torch.randn(4, 3, 16, 16, generator=generator)
    real = 3.0 * first + 4.0
    imag = 0.75 * first + 0.5 * second - 2.0
    x = torch.complex(real, imag).requires_grad_()
    norm = ComplexBatchNorm2d(3)

    output = norm(x)
    reduction_dims = (0, 2, 3)

    assert torch.allclose(output.real.mean(dim=reduction_dims), torch.zeros(3), atol=1e-5)
    assert torch.allclose(output.imag.mean(dim=reduction_dims), torch.zeros(3), atol=1e-5)
    assert torch.allclose(
        output.real.square().mean(dim=reduction_dims),
        torch.full((3,), 0.5),
        atol=2e-4,
    )
    assert torch.allclose(
        output.imag.square().mean(dim=reduction_dims),
        torch.full((3,), 0.5),
        atol=2e-4,
    )
    assert torch.allclose(
        (output.real * output.imag).mean(dim=reduction_dims),
        torch.zeros(3),
        atol=2e-4,
    )

    output.abs().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_complex_magnitude_max_pool_retains_full_complex_value():
    pool = ComplexMagnitudeMaxPool2d(2)
    x = torch.tensor(
        [[[[1 + 7j, 8 + 0j], [0 - 9j, -10 + 2j]]]],
        dtype=torch.complex64,
    )

    output = pool(x)

    assert output.shape == (1, 1, 1, 1)
    assert output.item() == pytest.approx(-10 + 2j)


def test_complex_magnitude_max_pool_is_phase_equivariant():
    pool = ComplexMagnitudeMaxPool2d(2)
    x = torch.complex(torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8))
    rotation = torch.polar(torch.tensor(1.0), torch.tensor(0.731))

    output = pool(x)
    rotated_output = pool(x * rotation)

    assert torch.allclose(rotated_output, output * rotation, atol=1e-6, rtol=1e-6)


def test_complex_magnitude_median_pool_retains_median_ranked_value():
    pool = ComplexMagnitudeMedianPool2d(2)
    x = torch.tensor(
        [[[[1 + 0j, 4 + 0j], [2 + 0j, 3 + 0j]]]],
        dtype=torch.complex64,
    )

    output = pool(x)

    assert output.shape == (1, 1, 1, 1)
    assert output.item() == pytest.approx(2 + 0j)


def test_complex_average_pool_averages_full_complex_values():
    pool = ComplexAveragePool2d(2)
    x = torch.tensor(
        [[[[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]]],
        dtype=torch.complex64,
    )

    output = pool(x)

    assert output.shape == (1, 1, 1, 1)
    assert output.item() == pytest.approx(2.5 + 2.5j)


@pytest.mark.parametrize(
    "pool",
    [ComplexMagnitudeMedianPool2d(2), ComplexAveragePool2d(2)],
)
def test_additional_complex_pools_are_phase_equivariant(pool):
    x = torch.complex(torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8))
    rotation = torch.polar(torch.tensor(1.0), torch.tensor(0.731))

    output = pool(x)
    rotated_output = pool(x * rotation)

    assert torch.allclose(rotated_output, output * rotation, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("pooling", ["max", "median", "average"])
def test_complex_model_builds_with_each_pooling_mode(pooling):
    model = build_model("complex", complex_pooling=pooling)

    assert model.pool1.__class__ in {
        ComplexMagnitudeMaxPool2d,
        ComplexMagnitudeMedianPool2d,
        ComplexAveragePool2d,
    }


def test_real_model_matches_complex_scalar_parameter_budget():
    expected = tuple(round(channel * 2**0.5) for channel in COMPLEX_CHANNELS)
    assert PARAMETER_MATCHED_REAL_CHANNELS == expected

    real_params = _trainable_params(build_model("real"))
    complex_params = _trainable_params(build_model("complex"))

    assert real_params / complex_params == pytest.approx(1.0, rel=0.01)


def test_kspace_model_uses_average_pooling_and_matches_parameter_budget():
    model = build_model("complex_kspace")
    kspace_params = _trainable_params(model)
    complex_params = _trainable_params(build_model("complex", complex_pooling="average"))
    real_params = _trainable_params(build_model("real"))

    assert isinstance(model, ComplexKSpaceCNN)
    assert all(
        isinstance(pool, ComplexAveragePool2d)
        for pool in (model.pool1, model.pool2, model.pool3)
    )
    assert kspace_params == complex_params
    assert real_params / kspace_params == pytest.approx(1.0, rel=0.01)

    x = torch.complex(torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32))
    model.eval()
    with torch.no_grad():
        output = model(x)
    assert output.shape == (2,)


def test_batchnorm_kspace_model_uses_complex_batch_norm_and_average_pooling():
    model = build_model("complex_kspace_batchnorm")

    assert isinstance(model, ComplexBatchNormKSpaceCNN)
    assert all(
        isinstance(pool, ComplexAveragePool2d)
        for pool in (model.pool1, model.pool2, model.pool3)
    )
    assert all(
        isinstance(norm, ComplexBatchNorm2d)
        for block in (model.block1, model.block2, model.block3, model.block4)
        for norm in (block.norm1, block.norm2)
    )

    x = torch.complex(torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32))
    output = model(x)

    assert output.shape == (2,)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("mode", "expected_type"),
    [
        ("complex_widely_linear", WidelyLinearComplexT2CNN),
        ("complex_modulus_gated", ModulusGatedT2CNN),
        ("complex_holographic_attention", HolographicAttentionT2CNN),
    ],
)
def test_specialized_complex_models_build_with_matched_parameter_budget(
    mode,
    expected_type,
):
    model = build_model(mode)
    real_params = _trainable_params(build_model("real"))
    model_params = _trainable_params(model)
    x = torch.complex(torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32))

    output = model(x)

    assert isinstance(model, expected_type)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
    assert real_params / model_params == pytest.approx(1.0, rel=0.01)
    output.mean().backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_widely_linear_model_uses_average_pooling_and_rms_norm():
    model = build_model("complex_widely_linear")

    assert all(
        isinstance(pool, ComplexAveragePool2d)
        for pool in (model.pool1, model.pool2, model.pool3)
    )
    assert all(
        isinstance(norm, ComplexRMSNorm2d)
        for block in (model.block1, model.block2, model.block3, model.block4)
        for norm in (block.norm1, block.norm2)
    )
    assert all(
        isinstance(convolution, WidelyLinearComplexConv2d)
        for block in (model.block1, model.block2, model.block3, model.block4)
        for convolution in (block.conv1, block.conv2)
    )


def test_modulus_gated_model_is_invariant_to_global_phase():
    x = torch.complex(torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32))
    model = build_model("complex_modulus_gated")
    model.eval()
    rotation = torch.polar(torch.tensor(1.0), torch.tensor(0.731))

    with torch.no_grad():
        output = model(x)
        rotated_output = model(x * rotation)

    assert torch.allclose(output, rotated_output, atol=1e-5, rtol=1e-5)


def test_holographic_model_uses_average_pooling_and_rms_norm():
    model = build_model("complex_holographic_attention")

    assert all(
        isinstance(pool, ComplexAveragePool2d)
        for pool in (model.pool1, model.pool2, model.pool3)
    )
    assert all(
        isinstance(norm, ComplexRMSNorm2d)
        for block in (model.block1, model.block2, model.block3, model.block4)
        for norm in (block.norm1, block.norm2)
    )
    assert isinstance(model.holographic_attention.norm, ComplexRMSNorm2d)


def test_holographic_model_is_invariant_to_global_phase():
    x = torch.complex(torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32))
    model = build_model("complex_holographic_attention")
    model.eval()
    rotation = torch.polar(torch.tensor(1.0), torch.tensor(1.234))

    with torch.no_grad():
        output = model(x)
        rotated_output = model(x * rotation)

    assert torch.allclose(output, rotated_output, atol=1e-5, rtol=1e-5)


def test_complex_model_is_invariant_to_global_phase():
    x = torch.complex(torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32))
    model = build_model("complex")
    model.eval()
    rotation = torch.polar(torch.tensor(1.0), torch.tensor(1.234))

    with torch.no_grad():
        output = model(x)
        rotated_output = model(x * rotation)

    assert output.shape == (2,)
    assert torch.allclose(output, rotated_output, atol=1e-5, rtol=1e-5)
