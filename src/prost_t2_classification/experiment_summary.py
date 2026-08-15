from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment_grid import ExperimentModelSpec, experiment_model_specs

MODEL_SPECS = {spec.model_key: spec for spec in experiment_model_specs()}
COMPLEX_MODELS = tuple(model for model in MODEL_SPECS if model != "real")
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

    Experiments with up to 22 paired seeds are enumerated exactly. Larger
    custom experiments use a deterministic one-million-draw Monte Carlo
    estimate so an override cannot allocate an exponentially large array.
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
    count: int,
    seed_base: int,
    model_specs: Mapping[str, ExperimentModelSpec] | None = None,
) -> None:
    if count < 1:
        raise ValueError("count must be positive.")

    selected_specs = MODEL_SPECS if model_specs is None else model_specs
    complex_models = tuple(model for model in selected_specs if model != "real")
    records: list[dict[str, object]] = []
    seeds = [seed_base + index + 1 for index in range(count)]
    for seed in seeds:
        for model, spec in selected_specs.items():
            model_dir = experiment_root / "phase2" / f"seed_{seed}" / "models" / model
            completion_marker = model_dir / "COMPLETE"
            if not completion_marker.is_file():
                raise FileNotFoundError(
                    f"Missing completion marker: {completion_marker}"
                )
            required = ("config.json", "history.csv", "threshold.json", "test_metrics.json")
            run_dirs = []
            for success in model_dir.glob("attempt_*/SUCCESS"):
                run_dirs.extend(
                    path.parent
                    for path in success.parent.glob("*/config.json")
                    if all((path.parent / name).is_file() for name in required)
                )
            if len(run_dirs) != 1:
                raise RuntimeError(
                    f"Expected one {model} run in {model_dir}; found {len(run_dirs)}"
                )
            records.append(_read_run(run_dirs[0], model=model, spec=spec, expected_seed=seed))

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
    delta_data: dict[str, pd.Series] = {}
    for model in complex_models:
        for metric in available:
            delta_data[_delta_column(metric, model)] = (
                wide[(metric, model)] - wide[(metric, "real")]
            )
    deltas = pd.DataFrame(delta_data, index=wide.index)
    deltas.reset_index().to_csv(experiment_root / "paired_deltas.csv", index=False)

    rng = np.random.default_rng(73191)
    paired_summary: list[dict[str, object]] = []
    for model in complex_models:
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
        "models": list(selected_specs),
        "comparisons": [f"{model}_minus_real" for model in complex_models],
        "primary_comparison": PRIMARY_COMPARISON,
        "exploratory_comparisons": [
            f"{model}_minus_real"
            for model in complex_models
            if f"{model}_minus_real" != PRIMARY_COMPARISON
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


def _read_run(
    run_dir: Path,
    *,
    model: str,
    spec: ExperimentModelSpec,
    expected_seed: int,
) -> dict[str, object]:
    required = ["config.json", "history.csv", "threshold.json", "test_metrics.json"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {', '.join(missing)} in {run_dir}")

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    expected_mode = spec.mode
    if config.get("mode") != expected_mode:
        raise RuntimeError(
            f"Expected mode={expected_mode} in {run_dir}; found {config.get('mode')!r}"
        )
    if int(config.get("seed", -1)) != expected_seed:
        raise RuntimeError(
            f"Expected seed={expected_seed} in {run_dir}; found {config.get('seed')!r}"
        )
    expected_pooling = spec.pooling
    expected_factors = {
        "model_key": spec.model_key,
        "complex_input_domain": spec.input_domain,
        "complex_pooling": spec.pooling,
        "complex_normalization": spec.normalization,
        "complex_convolution": spec.convolution,
        "complex_streams": spec.streams,
        "complex_interaction": spec.interaction,
        "complex_activation": spec.activation,
    }
    if expected_mode == "complex":
        for key, expected in expected_factors.items():
            if config.get(key) != expected:
                raise RuntimeError(
                    f"Expected {key}={expected} in {run_dir}; found {config.get(key)!r}"
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
        "input_domain": spec.input_domain,
        "pooling": expected_pooling,
        "normalization": spec.normalization,
        "convolution": spec.convolution,
        "streams": spec.streams,
        "interaction": spec.interaction,
        "activation": spec.activation,
        "trainable_parameters": int(config.get("trainable_parameters", 0)),
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
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    return parser


def _delta_column(metric: str, model: str) -> str:
    return f"{metric}_{model}_minus_real"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summarize_phase2(
        args.experiment_root,
        count=args.count,
        seed_base=args.seed_base,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
