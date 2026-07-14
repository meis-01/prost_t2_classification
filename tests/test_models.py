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


def test_real_model_uses_double_widths_for_complex_component_parity():
    assert PARAMETER_MATCHED_REAL_CHANNELS == tuple(channel * 2 for channel in COMPLEX_CHANNELS)

    real_params = _trainable_params(build_model("real", in_channels=1))
    modrelu_params = _trainable_params(build_model("complex", in_channels=1, complex_activation="modrelu"))

    assert real_params / modrelu_params == pytest.approx(2.0, rel=0.005)
