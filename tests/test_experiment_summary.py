import json

import numpy as np
import pandas as pd
import pytest

from prost_t2_classification.experiment_summary import (
    paired_sign_flip_pvalue,
    summarize_phase2,
)
from prost_t2_classification.experiment_grid import experiment_model_specs


def _write_run(run_dir, *, spec, seed, auc):
    run_dir.mkdir(parents=True)
    config = {
        "mode": spec.mode,
        "seed": seed,
        "model_key": spec.model_key,
        "complex_input_domain": spec.input_domain,
        "complex_pooling": spec.pooling,
        "complex_normalization": spec.normalization,
        "complex_convolution": spec.convolution,
        "complex_streams": spec.streams,
        "complex_interaction": spec.interaction,
        "complex_activation": spec.activation,
        "trainable_parameters": 1_680_000,
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    pd.DataFrame(
        [
            {"epoch": 0, "val_loss": 0.7, "val_auc": auc - 0.05},
            {"epoch": 1, "val_loss": 0.6, "val_auc": auc},
        ]
    ).to_csv(run_dir / "history.csv", index=False)
    threshold = {"validation_at_threshold": {"balanced_accuracy": auc}}
    (run_dir / "threshold.json").write_text(
        json.dumps(threshold), encoding="utf-8"
    )
    test_metrics = {
        "loss": 1.0 - auc,
        "auc": auc,
        "average_precision": auc - 0.01,
        "balanced_accuracy": auc - 0.02,
        "sensitivity": auc - 0.03,
        "specificity": auc - 0.04,
    }
    (run_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics), encoding="utf-8"
    )


def test_twenty_seed_sign_flip_test_is_exact():
    pvalue, method, permutations = paired_sign_flip_pvalue(np.ones(20))

    assert method == "exact"
    assert permutations == 2**20
    assert pvalue == pytest.approx(2 / 2**20)


def test_summarize_phase2_writes_all_pooling_comparisons(tmp_path):
    seed_base = 24000
    real_aucs = [0.60, 0.65, 0.70]
    selected_specs = {
        spec.model_key: spec for spec in experiment_model_specs()[:4]
    }
    offsets = {
        "real": 0.0,
        "complex_max": 0.05,
        "complex_median": 0.10,
        "complex_average": 0.02,
    }
    for index, real_auc in enumerate(real_aucs):
        seed = seed_base + index + 1
        for model, spec in selected_specs.items():
            model_dir = tmp_path / "phase2" / f"seed_{seed}" / "models" / model
            attempt_dir = model_dir / "attempt_1"
            attempt_dir.mkdir(parents=True)
            (model_dir / "COMPLETE").touch()
            (attempt_dir / "SUCCESS").touch()
            _write_run(
                attempt_dir / f"20260811_{model}",
                spec=spec,
                seed=seed,
                auc=real_auc + offsets[model],
            )

    summarize_phase2(
        tmp_path,
        count=len(real_aucs),
        seed_base=seed_base,
        model_specs=selected_specs,
    )

    metrics = pd.read_csv(tmp_path / "metrics_by_seed.csv")
    assert set(metrics["model"]) == {
        "real",
        "complex_max",
        "complex_median",
        "complex_average",
    }
    assert set(metrics["pooling"]) == {"none", "max", "median", "average"}

    deltas = pd.read_csv(tmp_path / "paired_deltas.csv")
    assert deltas["test_auc_complex_median_minus_real"].tolist() == pytest.approx(
        [0.10, 0.10, 0.10]
    )
    assert deltas["test_auc_complex_max_minus_real"].tolist() == pytest.approx(
        [0.05, 0.05, 0.05]
    )
    assert deltas["test_auc_complex_average_minus_real"].tolist() == pytest.approx(
        [0.02, 0.02, 0.02]
    )

    summary = pd.read_csv(tmp_path / "paired_summary.csv")
    primary = summary.loc[
        (summary["comparison"] == "complex_median_minus_real")
        & (summary["metric"] == "test_auc")
    ].iloc[0]
    assert bool(primary["is_primary_endpoint"])
    assert primary["mean_difference"] == pytest.approx(0.10)
    assert primary["complex_wins"] == 3
    assert primary["real_wins"] == 0
    assert primary["paired_sign_flip_p_value"] == pytest.approx(0.25)
    assert primary["sign_flip_method"] == "exact"
    assert primary["sign_flip_permutations"] == 8

    metadata = json.loads((tmp_path / "summary_metadata.json").read_text())
    assert metadata["primary_endpoint"] == "test_auc"
    assert metadata["primary_comparison"] == "complex_median_minus_real"
    assert metadata["exploratory_comparisons"] == [
        "complex_max_minus_real",
        "complex_average_minus_real",
    ]
    assert metadata["pilot_in_confirmatory_analysis"] is False
    assert metadata["confirmatory_seeds"] == [24001, 24002, 24003]
