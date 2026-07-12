import numpy as np
import pytest

from prost_t2_classification.train import TrainConfig, binary_metrics, resolve_in_channels, tune_threshold


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
