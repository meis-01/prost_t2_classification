import torch
import pytest

from prost_t2_classification.models import (
    COMPLEX_ACTIVATIONS,
    COMPLEX_CHANNELS,
    ModReLU,
    PARAMETER_MATCHED_REAL_CHANNELS,
    build_model,
)


def _trainable_params(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def test_complex_model_builds_with_each_activation():
    x = torch.complex(torch.randn(2, 5, 32, 32), torch.randn(2, 5, 32, 32))

    for activation in COMPLEX_ACTIVATIONS:
        model = build_model("complex", in_channels=5, complex_activation=activation)
        model.eval()
        with torch.no_grad():
            output = model(x)

        assert output.shape == (2,)


def test_modrelu_starts_without_negative_gate_bias():
    activation = ModReLU(4)

    assert torch.all(activation.bias == 0)


def test_real_model_matches_complex_scalar_parameter_budget():
    expected = tuple(round(channel * 2**0.5) for channel in COMPLEX_CHANNELS)
    assert PARAMETER_MATCHED_REAL_CHANNELS == expected

    real_params = _trainable_params(build_model("real", in_channels=4))
    complex_params = _trainable_params(build_model("complex", in_channels=4))

    assert real_params / complex_params == pytest.approx(1.0, rel=0.01)


def test_complex_model_is_invariant_to_global_phase():
    x = torch.complex(torch.randn(2, 4, 32, 32), torch.randn(2, 4, 32, 32))
    model = build_model("complex", in_channels=4)
    model.eval()
    rotation = torch.polar(torch.tensor(1.0), torch.tensor(1.234))

    with torch.no_grad():
        output = model(x)
        rotated_output = model(x * rotation)

    assert output.shape == (2,)
    assert torch.allclose(output, rotated_output, atol=1e-5, rtol=1e-5)
