import numpy as np
import pytest
import torch

from prost_t2_classification.train import (
    TrainConfig,
    _checkpoint_payload,
    binary_metrics,
    collect_epoch_outputs,
    resolve_in_channels,
    tune_threshold,
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


def test_training_infers_manifest_channel_count(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,channels\nsamples/a.npz,1\n", encoding="utf-8")

    config = TrainConfig(manifest=manifest, runs_dir=tmp_path / "runs", mode="real")
    assert resolve_in_channels(config) == 1

    mismatch = TrainConfig(manifest=manifest, runs_dir=tmp_path / "runs", mode="real", in_channels=5)
    with pytest.raises(ValueError, match="manifest samples contain 1 channel"):
        resolve_in_channels(mismatch)


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


def test_gradient_accumulation_steps_final_partial_group():
    model = torch.nn.Linear(2, 1)
    inputs = torch.ones(3, 2)
    targets = torch.tensor([0.0, 1.0, 1.0])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs, targets),
        batch_size=1,
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.weight.detach().clone()

    _, y_true, y_score = collect_epoch_outputs(
        model,
        loader,
        criterion,
        device=torch.device("cpu"),
        optimizer=optimizer,
        desc="test",
        gradient_accumulation_steps=2,
    )

    assert not torch.equal(before, model.weight.detach())
    assert y_true.tolist() == [0, 1, 1]
    assert y_score.shape == (3,)
