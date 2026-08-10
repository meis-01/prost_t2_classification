import torch
import pytest

from prost_t2_classification.models import (
    COMPLEX_CHANNELS,
    ComplexMagnitudeMaxPool2d,
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
