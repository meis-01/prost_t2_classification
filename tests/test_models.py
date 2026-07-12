import torch

from prost_t2_classification.models import COMPLEX_ACTIVATIONS, build_model


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


def test_real_model_is_parameter_matched_to_complex_backbone():
    real_params = _trainable_params(build_model("real", in_channels=5))
    crelu_params = _trainable_params(build_model("complex", in_channels=5, complex_activation="crelu"))
    cardioid_params = _trainable_params(build_model("complex", in_channels=5, complex_activation="cardioid"))
    modrelu_params = _trainable_params(build_model("complex", in_channels=5, complex_activation="modrelu"))

    assert real_params == crelu_params
    assert real_params == cardioid_params
    assert modrelu_params - real_params == 832
