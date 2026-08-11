#!/usr/bin/env python3
"""Run the paired real-vs-complex experiment sequentially on a laptop."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_EXPERIMENT_NAME = "laptop_maxpool_v1"
DEFAULT_PILOT_SEED = 10383
DEFAULT_PHASE2_SEED_BASE = 24000
DEFAULT_PHASE2_SEEDS = 3
DEFAULT_BOOTSTRAP_SEED = 73191


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _attempt_slug() -> str:
    return datetime.now(timezone.utc).strftime("attempt_%Y%m%dT%H%M%S_%fZ")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _default_threads() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exp-branch paired real/complex experiment locally, one seed at a time."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Validate the laptop, repository, and data.")
    _add_location_arguments(check, include_experiment=False)
    check.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    check.add_argument("--expected-branch", default="exp_local")
    check.set_defaults(func=check_command)

    run = subparsers.add_parser("run", help="Run or resume the local paired experiment.")
    _add_location_arguments(run, include_experiment=True)
    run.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    run.add_argument(
        "--complex-pooling",
        choices=("max", "median", "average"),
        default="max",
    )
    run.add_argument("--expected-branch", default="exp_local")
    run.add_argument("--pilot-seed", type=int, default=DEFAULT_PILOT_SEED)
    run.add_argument("--phase2-seeds", type=_positive_int, default=DEFAULT_PHASE2_SEEDS)
    run.add_argument("--phase2-seed-base", type=int, default=DEFAULT_PHASE2_SEED_BASE)
    run.add_argument("--epochs", type=_positive_int, default=20)
    run.add_argument("--batch-size", type=_positive_int, default=2)
    run.add_argument("--gradient-accumulation-steps", type=_positive_int, default=16)
    run.add_argument("--num-workers", type=_nonnegative_int, default=0)
    run.add_argument("--threads", type=_positive_int, default=_default_threads())
    run.add_argument("--patience", type=_positive_int, default=8)
    run.add_argument("--lr", type=float, default=1e-3)
    run.add_argument("--weight-decay", type=float, default=1e-4)
    run.set_defaults(func=run_command)

    summarize = subparsers.add_parser(
        "summarize", help="Regenerate result tables from completed phase-two seeds."
    )
    _add_location_arguments(summarize, include_experiment=True, include_manifest=False)
    summarize.set_defaults(func=summarize_command)
    return parser


def _add_location_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_experiment: bool,
    include_manifest: bool = True,
) -> None:
    if include_manifest:
        parser.add_argument(
            "--manifest",
            type=Path,
            default=REPO_ROOT / "data" / "manifest.csv",
        )
    if include_experiment:
        parser.add_argument(
            "--runs-root",
            type=Path,
            default=REPO_ROOT / "runs" / "prost_t2_experiments",
        )
        parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)


def _experiment_root(args: argparse.Namespace) -> Path:
    if not args.experiment_name.strip():
        raise ValueError("experiment name must not be empty")
    if Path(args.experiment_name).name != args.experiment_name:
        raise ValueError("experiment name must be a single directory name")
    return (args.runs_root / args.experiment_name).resolve()


def _import_dependencies():
    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import torch
        import tqdm
        from prost_t2_classification.dataset import validate_manifest
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing Python dependency {exc.name!r}. Activate the laptop virtual environment "
            'and run: python -m pip install --editable ".[dev]"'
        ) from exc
    return np, pd, sklearn, torch, tqdm, validate_manifest


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repository_info(expected_branch: str) -> dict[str, Any]:
    if shutil.which("git") is None:
        raise RuntimeError("git is not available")
    branch = _git_output("branch", "--show-current")
    if branch != expected_branch:
        raise RuntimeError(
            f"Expected branch {expected_branch!r}, but the current branch is {branch or 'detached'!r}."
        )
    status = _git_output("status", "--short")
    return {
        "branch": branch,
        "commit": _git_output("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def resolve_device(requested: str, torch) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid PyTorch device {requested!r}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch installation cannot use CUDA.")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested, but it is unavailable.")
    return str(device)


def validate_data(manifest_path: Path) -> dict[str, Any]:
    np, pd, _, _, _, validate_manifest = _import_dependencies()
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Put manifest.csv and samples/ under the data directory."
        )
    frame = pd.read_csv(manifest_path)
    validate_manifest(frame)
    missing = [
        str(value)
        for value in frame["path"]
        if not (manifest_path.parent / str(value)).is_file()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Manifest refers to {len(missing)} missing sample(s), including: {preview}"
        )

    first_sample = manifest_path.parent / str(frame.iloc[0]["path"])
    with np.load(first_sample) as sample:
        if "image_complex" not in sample:
            raise ValueError(f"{first_sample} does not contain image_complex")
        image = sample["image_complex"]
    if image.ndim != 3 or image.shape[0] != 4 or not np.iscomplexobj(image):
        raise ValueError(
            f"{first_sample} image_complex must be a complex array with shape (4, height, width)"
        )
    split_counts = (
        frame["data_split"].astype(str).str.strip().str.lower().value_counts().to_dict()
    )
    return {
        "manifest": str(manifest_path),
        "samples": len(frame),
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "sample_shape": list(image.shape),
    }


def environment_info(device: str, repo: dict[str, Any]) -> dict[str, Any]:
    np, pd, sklearn, torch, tqdm, _ = _import_dependencies()
    info: dict[str, Any] = {
        "recorded_utc": _utc_now(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "tqdm": tqdm.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cpu_count": os.cpu_count(),
        "repository": repo,
    }
    if torch.cuda.is_available():
        cuda_device = torch.device(device) if device.startswith("cuda") else torch.device("cuda")
        info["gpu"] = torch.cuda.get_device_name(cuda_device)
    return info


def validate_laptop(manifest: Path, device_request: str, expected_branch: str):
    if sys.version_info < (3, 10):
        raise RuntimeError(f"Python 3.10+ is required; found {platform.python_version()}")
    _, _, _, torch, _, _ = _import_dependencies()
    repo = repository_info(expected_branch)
    device = resolve_device(device_request, torch)
    data = validate_data(manifest)
    environment = environment_info(device, repo)
    return device, repo, data, environment


def check_command(args: argparse.Namespace) -> int:
    device, repo, data, environment = validate_laptop(
        args.manifest, args.device, args.expected_branch
    )
    print("Laptop check passed.")
    print(f"Branch: {repo['branch']} at {repo['commit'][:12]} (dirty={repo['dirty']})")
    print(f"Python: {environment['python']} ({environment['python_executable']})")
    print(f"PyTorch: {environment['torch']}; device: {device}")
    if environment.get("gpu"):
        print(f"GPU: {environment['gpu']}")
    print(
        f"Data: {data['samples']} samples; splits: {data['split_counts']}; "
        f"sample shape: {tuple(data['sample_shape'])}"
    )
    if repo["dirty"]:
        print("Warning: the worktree is dirty; its status will be recorded with the experiment.")
    return 0


def _configuration(args: argparse.Namespace, device: str) -> dict[str, Any]:
    return {
        "manifest": str(args.manifest.resolve()),
        "device": device,
        "complex_pooling": args.complex_pooling,
        "pilot_seed": args.pilot_seed,
        "phase2_seeds": args.phase2_seeds,
        "phase2_seed_base": args.phase2_seed_base,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_workers": args.num_workers,
        "threads": args.threads,
        "patience": args.patience,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
    }


def _prepare_experiment(
    root: Path,
    configuration: dict[str, Any],
    environment: dict[str, Any],
    data: dict[str, Any],
) -> None:
    config_path = root / "configuration.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != configuration:
            raise RuntimeError(
                f"{root} already has a different configuration. Use a new --experiment-name."
            )
    elif root.exists() and any(root.iterdir()):
        raise RuntimeError(f"Refusing to use non-empty experiment directory without config: {root}")
    else:
        root.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")

    metadata_path = root / "environment.json"
    if not metadata_path.exists():
        metadata = {**environment, "data": data}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _completed_attempts(seed_root: Path) -> list[Path]:
    if not seed_root.is_dir():
        return []
    return sorted(path.parent for path in seed_root.glob("attempt_*/COMPLETE"))


def _complex_run_pattern(pooling: str) -> str:
    if pooling == "max":
        return "*_complex_modrelu"
    return f"*_complex_modrelu_{pooling}_pool"


def _run_seed(
    *,
    phase: str,
    seed: int,
    root: Path,
    configuration: dict[str, Any],
) -> Path:
    seed_root = root / phase / f"seed_{seed}"
    completed = _completed_attempts(seed_root)
    if len(completed) > 1:
        raise RuntimeError(f"Found multiple completed attempts for seed {seed}: {completed}")
    if completed:
        print(f"Skipping completed {phase} seed {seed}: {completed[0]}")
        return completed[0]

    attempt = seed_root / _attempt_slug()
    logs = attempt / "logs"
    logs.mkdir(parents=True)
    command = [
        sys.executable,
        "-m",
        "prost_t2_classification",
        "--log-dir",
        str(logs),
        "train",
        "--manifest",
        configuration["manifest"],
        "--runs-dir",
        str(attempt),
        "--mode",
        "both",
        "--complex-pooling",
        configuration["complex_pooling"],
        "--device",
        configuration["device"],
        "--epochs",
        str(configuration["epochs"]),
        "--batch-size",
        str(configuration["batch_size"]),
        "--gradient-accumulation-steps",
        str(configuration["gradient_accumulation_steps"]),
        "--num-workers",
        str(configuration["num_workers"]),
        "--patience",
        str(configuration["patience"]),
        "--lr",
        str(configuration["learning_rate"]),
        "--weight-decay",
        str(configuration["weight_decay"]),
        "--seed",
        str(seed),
    ]
    run_info = {
        "phase": phase,
        "seed": seed,
        "started_utc": _utc_now(),
        "command": command,
        "python": sys.executable,
    }
    (attempt / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    child_environment = os.environ.copy()
    thread_count = str(configuration["threads"])
    child_environment.update(
        {
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
            "PYTHONUNBUFFERED": "1",
        }
    )

    print(f"\nRunning {phase} seed {seed} on {configuration['device']} ({attempt})")
    output_path = attempt / "console.log"
    return_code = -1
    with output_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                output.write(line)
                output.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=10)
            raise
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    if return_code != 0:
        (attempt / "FAILED").write_text(str(return_code), encoding="utf-8")
        raise RuntimeError(
            f"{phase} seed {seed} failed with exit code {return_code}; see {output_path}"
        )

    real_runs = [path for path in attempt.glob("*_real") if path.is_dir()]
    complex_pattern = _complex_run_pattern(configuration["complex_pooling"])
    complex_runs = [path for path in attempt.glob(complex_pattern) if path.is_dir()]
    if len(real_runs) != 1 or len(complex_runs) != 1:
        (attempt / "FAILED").write_text("invalid outputs", encoding="utf-8")
        raise RuntimeError(
            f"Expected one real and one complex output for seed {seed}; "
            f"found real={len(real_runs)}, complex={len(complex_runs)}"
        )
    for run_dir in (*real_runs, *complex_runs):
        for filename in ("history.csv", "threshold.json", "test_metrics.json"):
            if not (run_dir / filename).is_file():
                raise RuntimeError(f"Missing completed output: {run_dir / filename}")

    run_info["completed_utc"] = _utc_now()
    (attempt / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    (attempt / "COMPLETE").touch()
    return attempt


def run_command(args: argparse.Namespace) -> int:
    if args.lr <= 0:
        raise ValueError("learning rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight decay must be non-negative")
    device, _, data, environment = validate_laptop(
        args.manifest, args.device, args.expected_branch
    )
    root = _experiment_root(args)
    configuration = _configuration(args, device)
    _prepare_experiment(root, configuration, environment, data)

    _run_seed(
        phase="phase1",
        seed=args.pilot_seed,
        root=root,
        configuration=configuration,
    )
    for index in range(args.phase2_seeds):
        _run_seed(
            phase="phase2",
            seed=args.phase2_seed_base + index + 1,
            root=root,
            configuration=configuration,
        )
    summarize_experiment(root)
    print(f"\nExperiment complete. Results: {root}")
    return 0


def _only_completed_attempt(seed_root: Path) -> Path:
    completed = _completed_attempts(seed_root)
    if len(completed) != 1:
        raise RuntimeError(
            f"Expected one completed attempt in {seed_root}; found {len(completed)}"
        )
    return completed[0]


def summarize_experiment(root: Path) -> dict[str, Path]:
    np, pd, _, _, _, _ = _import_dependencies()
    root = root.resolve()
    config_path = root / "configuration.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing experiment configuration: {config_path}")
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    count = int(configuration["phase2_seeds"])
    seed_base = int(configuration["phase2_seed_base"])
    records: list[dict[str, Any]] = []

    for index in range(count):
        seed = seed_base + index + 1
        attempt = _only_completed_attempt(root / "phase2" / f"seed_{seed}")
        pooling = configuration.get("complex_pooling", "max")
        complex_suffix = _complex_run_pattern(pooling)
        model_patterns = {"real": "*_real", "complex": complex_suffix}
        for model, pattern in model_patterns.items():
            run_dirs = [path for path in attempt.glob(pattern) if path.is_dir()]
            if len(run_dirs) != 1:
                raise RuntimeError(
                    f"Expected one {model} run in {attempt}; found {len(run_dirs)}"
                )
            run_dir = run_dirs[0]
            test = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
            threshold = json.loads((run_dir / "threshold.json").read_text(encoding="utf-8"))
            history = pd.read_csv(run_dir / "history.csv")
            valid_auc = history["val_auc"].dropna()
            if valid_auc.empty:
                best_row = history.loc[history["val_loss"].idxmin()]
            else:
                best_row = history.loc[valid_auc.idxmax()]
            records.append(
                {
                    "seed": seed,
                    "model": model,
                    "epochs_completed": len(history),
                    "best_epoch": int(best_row["epoch"]) + 1,
                    "best_val_auc": float(best_row["val_auc"]),
                    "val_balanced_accuracy": float(
                        threshold["validation_at_threshold"]["balanced_accuracy"]
                    ),
                    **{f"test_{key}": float(value) for key, value in test.items()},
                    "run_dir": str(run_dir),
                }
            )

    metrics = pd.DataFrame(records).sort_values(["seed", "model"])
    output_paths = {
        "metrics": root / "metrics_by_seed.csv",
        "model_summary": root / "summary_by_model.csv",
        "deltas": root / "paired_deltas.csv",
        "paired_summary": root / "paired_summary.csv",
    }
    metrics.to_csv(output_paths["metrics"], index=False)

    value_columns = [
        "best_val_auc",
        "val_balanced_accuracy",
        "test_auc",
        "test_average_precision",
        "test_balanced_accuracy",
        "test_sensitivity",
        "test_specificity",
    ]
    available = [column for column in value_columns if column in metrics.columns]
    model_summary = metrics.groupby("model")[available].agg(["mean", "std"])
    model_summary.columns = [
        f"{metric}_{stat}" for metric, stat in model_summary.columns
    ]
    model_summary.reset_index().to_csv(output_paths["model_summary"], index=False)

    wide = metrics.pivot(index="seed", columns="model", values=available)
    deltas = pd.DataFrame(index=wide.index)
    for metric in available:
        deltas[f"{metric}_complex_minus_real"] = (
            wide[(metric, "complex")] - wide[(metric, "real")]
        )
    deltas.reset_index().to_csv(output_paths["deltas"], index=False)

    rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)
    paired_summary: list[dict[str, Any]] = []
    for column in deltas.columns:
        values = deltas[column].to_numpy(dtype=float)
        bootstrap = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
        paired_summary.append(
            {
                "metric": column.removesuffix("_complex_minus_real"),
                "n_pairs": len(values),
                "mean_complex_minus_real": values.mean(),
                "std_complex_minus_real": values.std(ddof=1) if len(values) > 1 else 0.0,
                "ci95_low": np.quantile(bootstrap, 0.025),
                "ci95_high": np.quantile(bootstrap, 0.975),
                "complex_wins": int((values > 0).sum()),
                "ties": int((values == 0).sum()),
                "real_wins": int((values < 0).sum()),
            }
        )
    pd.DataFrame(paired_summary).to_csv(output_paths["paired_summary"], index=False)
    (root / "PHASE2_COMPLETE").touch()
    print(f"Wrote summaries for {count} paired phase-two seeds to {root}")
    return output_paths


def summarize_command(args: argparse.Namespace) -> int:
    summarize_experiment(_experiment_root(args))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
