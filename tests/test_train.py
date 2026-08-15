import numpy as np
import pytest
import torch

from prost_t2_classification.train import (
    TrainConfig,
    _checkpoint_payload,
    binary_metrics,
    collect_epoch_outputs,
    train_both_models,
    tune_threshold,
    run_label_from_config,
)


def test_tune_threshold_maximizes_validation_balanced_accuracy():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.45, 0.8], dtype=np.float32)

    threshold = tune_threshold(y_true, y_score)
    metrics = binary_metrics(y_true, y_score, threshold=threshold)

    assert threshold == pytest.approx(0.45)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)


def test_tune_threshold_falls_back_to_half_for_single_class_validation():
    y_true = np.array([0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    assert tune_threshold(y_true, y_score) == 0.5


def test_checkpoint_payload_includes_optimizer_state():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    payload = _checkpoint_payload(
        model,
        optimizer,
        {"mode": "real"},
        epoch=3,
        score=0.7,
        best_score=0.8,
        bad_epochs=1,
    )

    assert "model_state" in payload
    assert "optimizer_state" in payload
    assert payload["config"] == {"mode": "real"}
    assert payload["epoch"] == 3
    assert payload["score"] == pytest.approx(0.7)
    assert payload["best_score"] == pytest.approx(0.8)
    assert payload["bad_epochs"] == 1
    assert "state" in payload["optimizer_state"]


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [
        ("max", "complex_modrelu"),
        ("median", "complex_modrelu_median_pool"),
        ("average", "complex_modrelu_average_pool"),
    ],
)
def test_complex_pooling_has_distinct_run_label(tmp_path, pooling, expected):
    config = TrainConfig(
        manifest=tmp_path / "manifest.csv",
        runs_dir=tmp_path / "runs",
        mode="complex",
        complex_pooling=pooling,
    )

    assert run_label_from_config(config) == expected


def test_kspace_model_has_distinct_average_pool_run_label(tmp_path):
    config = TrainConfig(
        manifest=tmp_path / "manifest.csv",
        runs_dir=tmp_path / "runs",
        mode="complex_kspace",
        complex_pooling="average",
    )

    assert run_label_from_config(config) == "complex_kspace_modrelu_average_pool"


def test_batchnorm_kspace_model_has_distinct_run_label(tmp_path):
    config = TrainConfig(
        manifest=tmp_path / "manifest.csv",
        runs_dir=tmp_path / "runs",
        mode="complex_kspace_batchnorm",
        complex_pooling="average",
    )

    assert run_label_from_config(config) == "complex_kspace_batchnorm_modrelu_average_pool"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "complex_widely_linear",
            "complex_widely_linear_modrelu_average_pool_rmsnorm",
        ),
        (
            "complex_modulus_gated",
            "complex_modulus_gated_modrelu_average_pool_rmsnorm",
        ),
    ],
)
def test_specialized_complex_models_have_distinct_run_labels(
    tmp_path,
    mode,
    expected,
):
    config = TrainConfig(
        manifest=tmp_path / "manifest.csv",
        runs_dir=tmp_path / "runs",
        mode=mode,
        complex_pooling="average",
    )

    assert run_label_from_config(config) == expected


@pytest.mark.parametrize(
    "mode",
    [
        "complex_kspace",
        "complex_kspace_batchnorm",
        "complex_widely_linear",
        "complex_modulus_gated",
    ],
)
def test_fixed_average_pool_models_reject_other_pooling(tmp_path, mode):
    with pytest.raises(ValueError, match="requires average"):
        TrainConfig(
            manifest=tmp_path / "manifest.csv",
            runs_dir=tmp_path / "runs",
            mode=mode,
            complex_pooling="max",
        )


def test_gradient_accumulation_steps_final_partial_group():
    reference_model = torch.nn.Linear(2, 1)
    accumulated_model = torch.nn.Linear(2, 1)
    accumulated_model.load_state_dict(reference_model.state_dict())
    inputs = torch.tensor([[0.5, -1.0]])
    targets = torch.tensor([1.0])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets),
        batch_size=1,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
    accumulated_optimizer = torch.optim.SGD(accumulated_model.parameters(), lr=0.1)

    collect_epoch_outputs(
        reference_model,
        loader,
        criterion,
        device=torch.device("cpu"),
        optimizer=reference_optimizer,
        desc="reference",
        gradient_accumulation_steps=1,
    )
    collect_epoch_outputs(
        accumulated_model,
        loader,
        criterion,
        device=torch.device("cpu"),
        optimizer=accumulated_optimizer,
        desc="accumulated",
        gradient_accumulation_steps=4,
    )

    for reference, accumulated in zip(reference_model.parameters(), accumulated_model.parameters()):
        assert torch.allclose(reference, accumulated)


def test_epoch_loss_is_weighted_by_sample_count():
    inputs = torch.tensor([[0.0], [0.0], [10.0]])
    targets = torch.zeros(3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets),
        batch_size=2,
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    loss, y_true, y_score = collect_epoch_outputs(
        torch.nn.Identity(),
        loader,
        criterion,
        device=torch.device("cpu"),
        optimizer=None,
        desc="weighted loss",
    )
    expected = criterion(inputs.flatten(), targets).item()

    assert loss == pytest.approx(expected)
    assert y_true.tolist() == [0, 0, 0]
    assert y_score.shape == (3,)


def test_prepared_npz_experiment_runs_both_models(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    rows = [
        (1, "training", 0),
        (2, "training", 1),
        (3, "validation", 0),
        (4, "validation", 1),
        (5, "test", 0),
        (6, "test", 1),
    ]
    manifest_lines = ["path,fastmri_pt_id,label,data_split,channels"]
    rng = np.random.default_rng(73191)
    for patient, split, label in rows:
        sample_name = f"patient_{patient}.npz"
        image = (
            rng.standard_normal((4, 32, 32))
            + 1j * rng.standard_normal((4, 32, 32))
        ).astype(np.complex64)
        np.savez_compressed(samples / sample_name, image_complex=image)
        manifest_lines.append(f"samples/{sample_name},{patient},{label},{split},4")

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    outputs = train_both_models(
        manifest,
        tmp_path / "runs",
        epochs=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        patience=1,
        seed=10383,
        num_workers=0,
        device="cpu",
    )

    assert set(outputs) == {"real", "complex"}
    for run_dir in outputs.values():
        assert (run_dir / "history.csv").is_file()
        assert (run_dir / "threshold.json").is_file()
        assert (run_dir / "test_metrics.json").is_file()
