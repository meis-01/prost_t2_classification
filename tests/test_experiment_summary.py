import json

import numpy as np
import pandas as pd
import pytest

from prost_t2_classification.experiment_summary import (
    paired_sign_flip_pvalue,
    summarize_phase2,
)


def _write_run(run_dir, *, mode, seed, auc, pooling="none"):
    run_dir.mkdir()
    config = {
        "mode": mode,
        "seed": seed,
        "complex_pooling": pooling,
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


def test_default_twenty_seed_sign_flip_test_is_exact():
    pvalue, method, permutations = paired_sign_flip_pvalue(np.ones(20))

    assert method == "exact"
    assert permutations == 2**20
    assert pvalue == pytest.approx(2 / 2**20)


def test_summarize_phase2_writes_all_pooling_comparisons(tmp_path):
    array_job = "9876"
    seed_base = 24000
    real_aucs = [0.60, 0.65, 0.70]
    for index, real_auc in enumerate(real_aucs):
        seed = seed_base + index + 1
        job_dir = (
            tmp_path
            / "phase2"
            / f"seed_{seed}"
            / f"job_{array_job}_{index}"
        )
        job_dir.mkdir(parents=True)
        for marker in (
            "REAL_COMPLETE",
            "COMPLEX_MAX_COMPLETE",
            "COMPLEX_MEDIAN_COMPLETE",
            "COMPLEX_AVERAGE_COMPLETE",
        ):
            (job_dir / marker).touch()
        _write_run(job_dir / "20260811_real", mode="real", seed=seed, auc=real_auc)
        _write_run(
            job_dir / "20260811_complex_modrelu",
            mode="complex",
            seed=seed,
            auc=real_auc + 0.05,
            pooling="max",
        )
        _write_run(
            job_dir / "20260811_complex_modrelu_median_pool",
            mode="complex",
            seed=seed,
            auc=real_auc + 0.10,
            pooling="median",
        )
        _write_run(
            job_dir / "20260811_complex_modrelu_average_pool",
            mode="complex",
            seed=seed,
            auc=real_auc + 0.02,
            pooling="average",
        )

    summarize_phase2(
        tmp_path,
        array_job=array_job,
        count=len(real_aucs),
        seed_base=seed_base,
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
