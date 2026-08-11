from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.run_local_experiment import _prepare_experiment, summarize_experiment


def test_prepare_experiment_rejects_configuration_drift(tmp_path):
    root = tmp_path / "experiment"
    _prepare_experiment(root, {"epochs": 2}, {"python": "test"}, {"samples": 6})

    with pytest.raises(RuntimeError, match="different configuration"):
        _prepare_experiment(root, {"epochs": 3}, {"python": "test"}, {"samples": 6})


@pytest.mark.parametrize(
    ("pooling", "complex_suffix"),
    [
        (None, "complex_modrelu"),
        ("median", "complex_modrelu_median_pool"),
        ("average", "complex_modrelu_average_pool"),
    ],
)
def test_summarize_experiment_writes_paired_tables(
    tmp_path, pooling, complex_suffix
):
    root = tmp_path / "experiment"
    root.mkdir()
    configuration = {"phase2_seeds": 2, "phase2_seed_base": 100}
    if pooling is not None:
        configuration["complex_pooling"] = pooling
    (root / "configuration.json").write_text(
        json.dumps(configuration),
        encoding="utf-8",
    )

    for seed in (101, 102):
        attempt = root / "phase2" / f"seed_{seed}" / "attempt_test"
        attempt.mkdir(parents=True)
        (attempt / "COMPLETE").touch()
        for model, suffix, offset in (
            ("real", "real", 0.0),
            ("complex", complex_suffix, 0.1),
        ):
            run_dir = attempt / f"run_{suffix}"
            run_dir.mkdir()
            pd.DataFrame(
                {
                    "epoch": [0.0, 1.0],
                    "val_loss": [0.8, 0.7],
                    "val_auc": [0.6 + offset, 0.7 + offset],
                }
            ).to_csv(run_dir / "history.csv", index=False)
            (run_dir / "threshold.json").write_text(
                json.dumps(
                    {
                        "validation_at_threshold": {
                            "balanced_accuracy": 0.65 + offset,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "test_metrics.json").write_text(
                json.dumps(
                    {
                        "auc": 0.7 + offset,
                        "average_precision": 0.68 + offset,
                        "balanced_accuracy": 0.66 + offset,
                        "sensitivity": 0.64 + offset,
                        "specificity": 0.68 + offset,
                    }
                ),
                encoding="utf-8",
            )

    outputs = summarize_experiment(root)

    assert all(path.is_file() for path in outputs.values())
    assert (root / "PHASE2_COMPLETE").is_file()
    metrics = pd.read_csv(outputs["metrics"])
    assert len(metrics) == 4
    deltas = pd.read_csv(outputs["deltas"])
    assert deltas["test_auc_complex_minus_real"].tolist() == pytest.approx([0.1, 0.1])
    paired = pd.read_csv(outputs["paired_summary"])
    auc = paired.loc[paired["metric"] == "test_auc"].iloc[0]
    assert auc["n_pairs"] == 2
    assert auc["mean_complex_minus_real"] == pytest.approx(0.1)
    assert auc["complex_wins"] == 2
