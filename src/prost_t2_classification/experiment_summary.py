from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_SPECS = {
    "real": {"pattern": "*_real", "mode": "real", "pooling": "none"},
    "complex_max": {
        "pattern": "*_complex_modrelu",
        "mode": "complex",
        "pooling": "max",
    },
    "complex_median": {
        "pattern": "*_complex_modrelu_median_pool",
        "mode": "complex",
        "pooling": "median",
    },
    "complex_average": {
        "pattern": "*_complex_modrelu_average_pool",
        "mode": "complex",
        "pooling": "average",
    },
}
COMPLEX_MODELS = ("complex_max", "complex_median", "complex_average")
VALUE_COLUMNS = [
    "best_val_auc",
    "val_balanced_accuracy",
    "test_auc",
    "test_average_precision",
    "test_balanced_accuracy",
    "test_sensitivity",
    "test_specificity",
]
PRIMARY_ENDPOINT = "test_auc"
PRIMARY_COMPARISON = "complex_median_minus_real"


def paired_sign_flip_pvalue(values: np.ndarray) -> tuple[float, str, int]:
    """Return a two-sided paired sign-flip p-value for the mean difference.

    The default 20-seed experiment is enumerated exactly. Larger custom
    experiments use a deterministic one-million-draw Monte Carlo estimate so
    an override cannot allocate an exponentially large array.
    """

    differences = np.asarray(values, dtype=float)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("Paired differences must be a non-empty one-dimensional array.")
    if not np.isfinite(differences).all():
        raise ValueError("Paired differences must all be finite.")

    observed = abs(float(differences.mean()))
    tolerance = np.finfo(float).eps * max(1.0, observed) * 8
    count = differences.size

    if count <= 22:
        permutations = 1 << count
        extreme = 0
        chunk_size = 1 << min(count, 16)
        for start in range(0, permutations, chunk_size):
            stop = min(start + chunk_size, permutations)
            masks = np.arange(start, stop, dtype=np.uint64)
            signed_sums = np.zeros(stop - start, dtype=float)
            for bit, value in enumerate(differences):
                signs = np.where(((masks >> bit) & 1) == 1, 1.0, -1.0)
                signed_sums += signs * value
            extreme += int(
                (np.abs(signed_sums / count) >= observed - tolerance).sum()
            )
        return extreme / permutations, "exact", permutations

    draws = 1_000_000
    rng = np.random.default_rng(73192)
    extreme = 0
    completed = 0
    while completed < draws:
        chunk = min(10_000, draws - completed)
        signs = rng.integers(0, 2, size=(chunk, count), dtype=np.int8) * 2 - 1
        randomized_means = signs @ differences / count
        extreme += int((np.abs(randomized_means) >= observed - tolerance).sum())
        completed += chunk
    return (extreme + 1) / (draws + 1), "monte_carlo", draws


def summarize_phase2(
    experiment_root: Path,
    *,
    array_job: str,
    count: int,
    seed_base: int,
) -> None:
    if count < 1:
        raise ValueError("count must be positive.")

    records: list[dict[str, object]] = []
    seeds = [seed_base + index + 1 for index in range(count)]
    for index, seed in enumerate(seeds):
        job_dir = (
            experiment_root
            / "phase2"
            / f"seed_{seed}"
            / f"job_{array_job}_{index}"
        )
        for model, spec in MODEL_SPECS.items():
            completion_marker = job_dir / f"{model.upper()}_COMPLETE"
            if not completion_marker.is_file():
                raise FileNotFoundError(
                    f"Missing completion marker: {completion_marker}"
                )
            run_dirs = [
                path for path in job_dir.glob(str(spec["pattern"])) if path.is_dir()
            ]
            if len(run_dirs) != 1:
                raise RuntimeError(
                    f"Expected one {model} run in {job_dir}; found {len(run_dirs)}"
                )
            records.append(_read_run(run_dirs[0], model=model, expected_seed=seed))

    metrics = pd.DataFrame(records).sort_values(["seed", "model"])
    metrics.to_csv(experiment_root / "metrics_by_seed.csv", index=False)

    available = [column for column in VALUE_COLUMNS if column in metrics.columns]
    missing = sorted(set(VALUE_COLUMNS) - set(available))
    if missing:
        raise RuntimeError(f"Missing required summary metrics: {', '.join(missing)}")

    model_summary = metrics.groupby("model")[available].agg(["mean", "std"])
    model_summary.columns = [
        f"{metric}_{stat}" for metric, stat in model_summary.columns
    ]
    model_summary.reset_index().to_csv(
        experiment_root / "summary_by_model.csv", index=False
    )

    wide = metrics.pivot(index="seed", columns="model", values=available)
    deltas = pd.DataFrame(index=wide.index)
    for model in COMPLEX_MODELS:
        for metric in available:
            deltas[_delta_column(metric, model)] = (
                wide[(metric, model)] - wide[(metric, "real")]
            )
    deltas.reset_index().to_csv(experiment_root / "paired_deltas.csv", index=False)

    rng = np.random.default_rng(73191)
    paired_summary: list[dict[str, object]] = []
    for model in COMPLEX_MODELS:
        comparison = f"{model}_minus_real"
        for metric in available:
            column = _delta_column(metric, model)
            values = deltas[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite paired values for {column}.")
            bootstrap = rng.choice(
                values, size=(10_000, len(values)), replace=True
            ).mean(axis=1)
            is_primary = (
                comparison == PRIMARY_COMPARISON and metric == PRIMARY_ENDPOINT
            )
            if is_primary:
                pvalue, test_method, permutations = paired_sign_flip_pvalue(values)
            else:
                pvalue, test_method, permutations = np.nan, "", 0
            paired_summary.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "is_primary_endpoint": is_primary,
                    "n_pairs": len(values),
                    "mean_difference": values.mean(),
                    "std_difference": values.std(ddof=1),
                    "ci95_low": np.quantile(bootstrap, 0.025),
                    "ci95_high": np.quantile(bootstrap, 0.975),
                    "complex_wins": int((values > 0).sum()),
                    "ties": int((values == 0).sum()),
                    "real_wins": int((values < 0).sum()),
                    "paired_sign_flip_p_value": pvalue,
                    "sign_flip_method": test_method,
                    "sign_flip_permutations": permutations,
                }
            )
    pd.DataFrame(paired_summary).to_csv(
        experiment_root / "paired_summary.csv", index=False
    )

    metadata = {
        "models": list(MODEL_SPECS),
        "comparisons": [f"{model}_minus_real" for model in COMPLEX_MODELS],
        "primary_comparison": PRIMARY_COMPARISON,
        "exploratory_comparisons": [
            "complex_max_minus_real",
            "complex_average_minus_real",
        ],
        "primary_endpoint": PRIMARY_ENDPOINT,
        "secondary_endpoints": [
            "test_average_precision",
            "test_balanced_accuracy",
            "test_sensitivity",
            "test_specificity",
        ],
        "pilot_in_confirmatory_analysis": False,
        "confirmatory_seeds": seeds,
        "bootstrap_resamples": 10_000,
        "inference_scope": "training-seed variability on the fixed test set",
    }
    (experiment_root / "summary_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote phase-two summaries for {count} paired seeds to {experiment_root}")


def _read_run(run_dir: Path, *, model: str, expected_seed: int) -> dict[str, object]:
    required = ["config.json", "history.csv", "threshold.json", "test_metrics.json"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {', '.join(missing)} in {run_dir}")

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    spec = MODEL_SPECS[model]
    expected_mode = str(spec["mode"])
    if config.get("mode") != expected_mode:
        raise RuntimeError(
            f"Expected mode={expected_mode} in {run_dir}; found {config.get('mode')!r}"
        )
    if int(config.get("seed", -1)) != expected_seed:
        raise RuntimeError(
            f"Expected seed={expected_seed} in {run_dir}; found {config.get('seed')!r}"
        )
    expected_pooling = str(spec["pooling"])
    if expected_mode == "complex" and config.get("complex_pooling") != expected_pooling:
        raise RuntimeError(
            f"Expected {expected_pooling} pooling in {run_dir}; "
            f"found {config.get('complex_pooling')!r}"
        )

    test = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    threshold = json.loads((run_dir / "threshold.json").read_text(encoding="utf-8"))
    history = pd.read_csv(run_dir / "history.csv")
    if history.empty:
        raise RuntimeError(f"Empty training history: {run_dir / 'history.csv'}")
    if history["val_auc"].notna().any():
        best_row = history.loc[history["val_auc"].idxmax()]
    else:
        best_row = history.loc[history["val_loss"].idxmin()]

    return {
        "seed": expected_seed,
        "model": model,
        "pooling": expected_pooling,
        "epochs_completed": len(history),
        "best_epoch": int(best_row["epoch"]) + 1,
        "best_val_auc": float(best_row["val_auc"]),
        "val_balanced_accuracy": float(
            threshold["validation_at_threshold"]["balanced_accuracy"]
        ),
        **{f"test_{key}": float(value) for key, value in test.items()},
        "run_dir": str(run_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate paired real and complex-pooling seed runs."
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--array-job", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    return parser


def _delta_column(metric: str, model: str) -> str:
    return f"{metric}_{model}_minus_real"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summarize_phase2(
        args.experiment_root,
        array_job=args.array_job,
        count=args.count,
        seed_base=args.seed_base,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
