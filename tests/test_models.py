import torch
import pytest

from prost_t2_classification.models import (
    COMPLEX_ACTIVATIONS,
    COMPLEX_CHANNELS,
    COMPLEX_VARIANTS,
    InputPhaseGate,
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


def test_phase_gate_starts_as_magnitude_only():
    gate = InputPhaseGate(2)
    x = torch.complex(torch.randn(3, 2, 8, 8), torch.randn(3, 2, 8, 8))

    y = gate(x)

    assert torch.all(gate.phase_scale == 0)
    assert torch.allclose(y.real, torch.abs(x), atol=1e-6)
    assert torch.allclose(y.imag, torch.zeros_like(y.imag), atol=1e-6)


def test_real_model_uses_double_widths_for_complex_component_parity():
    assert PARAMETER_MATCHED_REAL_CHANNELS == tuple(channel * 2 for channel in COMPLEX_CHANNELS)

    real_params = _trainable_params(build_model("real", in_channels=1))
    modrelu_params = _trainable_params(build_model("complex", in_channels=1, complex_activation="modrelu"))

    assert real_params / modrelu_params == pytest.approx(2.0, rel=0.005)


def test_widely_linear_phase_variant_builds_near_real_scalar_budget():
    x = torch.complex(torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32))
    model = build_model(
        "complex",
        in_channels=1,
        complex_activation="modrelu",
        complex_variant="widely_linear_phase",
    )
    model.eval()

    with torch.no_grad():
        output = model(x)

    real_params = _trainable_params(build_model("real", in_channels=1))
    wide_params = _trainable_params(model)
    assert output.shape == (2,)
    assert real_params / wide_params == pytest.approx(1.0, rel=0.01)


def test_hybrid_variant_builds_near_real_scalar_budget():
    assert "hybrid" in COMPLEX_VARIANTS
    x = torch.complex(torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32))
    model = build_model(
        "complex",
        in_channels=1,
        complex_activation="modrelu",
        complex_variant="hybrid",
    )
    model.eval()

    with torch.no_grad():
        output = model(x)

    real_params = _trainable_params(build_model("real", in_channels=1))
    hybrid_params = _trainable_params(model)
    assert output.shape == (2,)
    assert real_params / hybrid_params == pytest.approx(1.0, rel=0.01)
