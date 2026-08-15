import torch
import pytest

from prost_t2_classification.models import (
    COMPLEX_CHANNELS,
    ComplexAveragePool2d,
    CardioidActivation,
    ComplexReLU,
    ComplexMagnitudeMaxPool2d,
    ComplexMagnitudeMedianPool2d,
    MagnitudeGatedSiLU,
    ModReLU,
    PARAMETER_MATCHED_REAL_CHANNELS,
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


def test_magnitude_gated_silu_is_phase_equivariant():
    activation = MagnitudeGatedSiLU(3)
    x = torch.complex(torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8))
    rotation = torch.polar(torch.tensor(1.0), torch.tensor(0.731))

    assert torch.allclose(
        activation(x * rotation), activation(x) * rotation, atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("activation", [ComplexReLU(3), CardioidActivation(3)])
def test_phase_selective_activations_preserve_shape_and_complex_dtype(activation):
    x = torch.complex(torch.randn(2, 3, 8, 8), torch.randn(2, 3, 8, 8))

    output = activation(x)

    assert output.shape == x.shape
    assert torch.is_complex(output)


@pytest.mark.parametrize(
    ("normalization", "convolution", "streams", "interaction", "activation"),
    [
        ("batchnorm", "standard", "complex_only", "none", "crelu"),
        ("rms", "widely_linear", "complex_only", "holographic", "cardioid"),
        ("rms", "standard", "dual", "none", "magnitude_silu"),
        ("batchnorm", "standard", "dual", "modulus_gate", "modrelu"),
        ("batchnorm", "widely_linear", "dual", "holographic", "crelu"),
    ],
)
def test_grid_architecture_families_build_forward_and_match_parameter_budget(
    normalization, convolution, streams, interaction, activation
):
    model = build_model(
        "complex",
        complex_pooling="average",
        complex_normalization=normalization,
        complex_convolution=convolution,
        complex_streams=streams,
        complex_interaction=interaction,
        complex_activation=activation,
    )
    model.eval()
    x = torch.complex(torch.randn(2, 4, 16, 16), torch.randn(2, 4, 16, 16))

    with torch.no_grad():
        output = model(x)

    real_parameters = _trainable_params(build_model("real"))
    assert output.shape == (2,)
    assert _trainable_params(model) / real_parameters == pytest.approx(1.0, rel=0.01)


def test_complex_only_modulus_gate_is_rejected():
    with pytest.raises(ValueError, match="requires dual streams"):
        build_model(
            "complex",
            complex_streams="complex_only",
            complex_interaction="modulus_gate",
        )
