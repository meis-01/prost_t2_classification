#!/usr/bin/env bash
set -Eeuo pipefail

# One-command CPU/Slurm experiment.
#
# Expected repository layout after cloning branch "exp":
#   data/manifest.csv
#   data/samples/*.npz
#
# Run from anywhere inside the clone:
#   bash scripts/run_cluster_experiment.sh
#
# Optional overrides, for example:
#   PARTITION=general CPUS=20 MAX_PARALLEL=6 bash scripts/run_cluster_experiment.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
MANIFEST="${MANIFEST:-${DATA_DIR}/manifest.csv}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-maxpool_v1}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${REPO_ROOT}/runs/${EXPERIMENT_NAME}}"

PARTITION="${PARTITION:-work}"
CPUS="${CPUS:-16}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

PILOT_SEED="${PILOT_SEED:-10383}"
PHASE2_SEEDS="${PHASE2_SEEDS:-20}"
PHASE2_SEED_BASE="${PHASE2_SEED_BASE:-24000}"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PATIENCE="${PATIENCE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-exp}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer; got ${value}."
}

check_configuration() {
    require_positive_integer CPUS "${CPUS}"
    require_positive_integer MAX_PARALLEL "${MAX_PARALLEL}"
    require_positive_integer PHASE2_SEEDS "${PHASE2_SEEDS}"
    require_positive_integer EPOCHS "${EPOCHS}"
    require_positive_integer BATCH_SIZE "${BATCH_SIZE}"
    require_positive_integer GRADIENT_ACCUMULATION_STEPS "${GRADIENT_ACCUMULATION_STEPS}"
    require_positive_integer PATIENCE "${PATIENCE}"
    [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || die "NUM_WORKERS must be a non-negative integer."
    (( NUM_WORKERS < CPUS )) || die "NUM_WORKERS must be smaller than CPUS."
}

check_repository() {
    command -v git >/dev/null 2>&1 || die "git is not available."
    local branch
    branch="$(git -C "${REPO_ROOT}" branch --show-current)"
    [[ "${branch}" == "${EXPECTED_BRANCH}" ]] || die "Expected branch ${EXPECTED_BRANCH}; current branch is ${branch:-detached}."
    if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
        git -C "${REPO_ROOT}" diff --quiet || die "Tracked files have unstaged changes."
        git -C "${REPO_ROOT}" diff --cached --quiet || die "Tracked files have staged changes."
    fi
}

ensure_environment() {
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "${PYTHON_BIN} is not available. Set PYTHON_BIN to a cluster Python 3 executable."
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
        "${VENV_DIR}/bin/python" -m pip install --upgrade pip
        "${VENV_DIR}/bin/python" -m pip install torch --index-url "${TORCH_INDEX_URL}"
        "${VENV_DIR}/bin/python" -m pip install numpy pandas scikit-learn tqdm
    fi
    "${VENV_DIR}/bin/python" -m pip install --no-deps --editable "${REPO_ROOT}"
    "${VENV_DIR}/bin/python" -c 'import numpy, pandas, sklearn, torch, tqdm, prost_t2_classification'
}

validate_data() {
    [[ -f "${MANIFEST}" ]] || die "Missing ${MANIFEST}. Copy manifest.csv and samples/ into ${DATA_DIR}."
    [[ -d "${DATA_DIR}/samples" ]] || die "Missing ${DATA_DIR}/samples."
    "${VENV_DIR}/bin/python" - "${MANIFEST}" <<'PY'
from pathlib import Path
import sys

import pandas as pd

manifest_path = Path(sys.argv[1])
frame = pd.read_csv(manifest_path)
required = {"path", "label", "data_split", "channels"}
missing_columns = sorted(required.difference(frame.columns))
if missing_columns:
    raise SystemExit(f"Manifest is missing columns: {missing_columns}")
if frame.empty:
    raise SystemExit("Manifest is empty")
missing_files = [value for value in frame["path"] if not (manifest_path.parent / str(value)).is_file()]
if missing_files:
    preview = ", ".join(map(str, missing_files[:5]))
    raise SystemExit(f"Manifest refers to {len(missing_files)} missing sample(s), including: {preview}")
split_counts = frame["data_split"].astype(str).str.lower().value_counts().to_dict()
channel_counts = sorted({int(value) for value in frame["channels"].dropna()})
if channel_counts != [4]:
    raise SystemExit(f"Expected exactly four channels in every sample; found {channel_counts}")
print(f"Validated {len(frame)} samples; split counts: {split_counts}; channels: 4")
PY
}

export_job_environment() {
    export REPO_ROOT DATA_DIR MANIFEST VENV_DIR EXPERIMENT_NAME EXPERIMENT_ROOT
    export PARTITION CPUS MEMORY TIME_LIMIT MAX_PARALLEL
    export PILOT_SEED PHASE2_SEEDS PHASE2_SEED_BASE
    export EPOCHS BATCH_SIZE GRADIENT_ACCUMULATION_STEPS NUM_WORKERS PATIENCE
    export LEARNING_RATE WEIGHT_DECAY PYTHON_BIN TORCH_INDEX_URL EXPECTED_BRANCH
}

write_submission_metadata() {
    local commit
    commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    {
        printf 'git_commit=%s\n' "${commit}"
        printf 'manifest=%s\n' "${MANIFEST}"
        printf 'partition=%s\n' "${PARTITION}"
        printf 'cpus=%s\n' "${CPUS}"
        printf 'memory=%s\n' "${MEMORY}"
        printf 'time_limit=%s\n' "${TIME_LIMIT}"
        printf 'pilot_seed=%s\n' "${PILOT_SEED}"
        printf 'phase2_seeds=%s\n' "${PHASE2_SEEDS}"
        printf 'phase2_seed_base=%s\n' "${PHASE2_SEED_BASE}"
        printf 'epochs=%s\n' "${EPOCHS}"
        printf 'batch_size=%s\n' "${BATCH_SIZE}"
        printf 'gradient_accumulation_steps=%s\n' "${GRADIENT_ACCUMULATION_STEPS}"
        printf 'num_workers=%s\n' "${NUM_WORKERS}"
        printf 'patience=%s\n' "${PATIENCE}"
        printf 'learning_rate=%s\n' "${LEARNING_RATE}"
        printf 'weight_decay=%s\n' "${WEIGHT_DECAY}"
    } > "${EXPERIMENT_ROOT}/configuration.txt"
    "${VENV_DIR}/bin/python" -m pip freeze > "${EXPERIMENT_ROOT}/environment.txt"
}

submit_experiment() {
    command -v sbatch >/dev/null 2>&1 || die "sbatch is not available; run this command on a Slurm login node."
    check_configuration
    check_repository
    [[ ! -e "${EXPERIMENT_ROOT}/submission.txt" ]] || die "${EXPERIMENT_ROOT} was already submitted. Set a new EXPERIMENT_NAME."

    ensure_environment
    validate_data
    mkdir -p "${EXPERIMENT_ROOT}/slurm" "${EXPERIMENT_ROOT}/phase1" "${EXPERIMENT_ROOT}/phase2"
    write_submission_metadata
    export_job_environment

    local pilot_job array_job summary_job array_last array_spec
    pilot_job="$(sbatch --parsable \
        --job-name=prost-pilot \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --chdir="${REPO_ROOT}" \
        --output="${EXPERIMENT_ROOT}/slurm/phase1_%j.out" \
        --error="${EXPERIMENT_ROOT}/slurm/phase1_%j.err" \
        --export=ALL \
        "${SCRIPT_PATH}" __worker phase1)"
    pilot_job="${pilot_job%%;*}"

    array_last=$((PHASE2_SEEDS - 1))
    array_spec="0-${array_last}%${MAX_PARALLEL}"
    array_job="$(sbatch --parsable \
        --job-name=prost-seeds \
        --partition="${PARTITION}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --array="${array_spec}" \
        --dependency="afterok:${pilot_job}" \
        --chdir="${REPO_ROOT}" \
        --output="${EXPERIMENT_ROOT}/slurm/phase2_%A_%a.out" \
        --error="${EXPERIMENT_ROOT}/slurm/phase2_%A_%a.err" \
        --export=ALL \
        "${SCRIPT_PATH}" __worker phase2)"
    array_job="${array_job%%;*}"

    summary_job="$(sbatch --parsable \
        --job-name=prost-summary \
        --partition="${PARTITION}" \
        --cpus-per-task=1 \
        --mem=4G \
        --time=00:30:00 \
        --dependency="afterok:${array_job}" \
        --chdir="${REPO_ROOT}" \
        --output="${EXPERIMENT_ROOT}/slurm/summary_%j.out" \
        --error="${EXPERIMENT_ROOT}/slurm/summary_%j.err" \
        --export=ALL \
        "${SCRIPT_PATH}" __summarize "${array_job}")"
    summary_job="${summary_job%%;*}"

    {
        printf 'phase1_job=%s\n' "${pilot_job}"
        printf 'phase2_array_job=%s\n' "${array_job}"
        printf 'summary_job=%s\n' "${summary_job}"
    } | tee "${EXPERIMENT_ROOT}/submission.txt"

    printf '\nPhase two contains %s paired seeds and starts only if phase one succeeds.\n' "${PHASE2_SEEDS}"
    printf 'Monitor with: squeue -j %s,%s,%s\n' "${pilot_job}" "${array_job}" "${summary_job}"
    printf 'Results will be written under: %s\n' "${EXPERIMENT_ROOT}"
}

run_worker() {
    local phase="${1:?worker phase is required}"
    local seed task_index job_token run_dir torch_threads exit_status
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "The worker must run inside a Slurm allocation."
    [[ -x "${VENV_DIR}/bin/python" ]] || die "Missing experiment environment at ${VENV_DIR}."

    case "${phase}" in
        phase1)
            seed="${PILOT_SEED}"
            job_token="${SLURM_JOB_ID}"
            ;;
        phase2)
            task_index="${SLURM_ARRAY_TASK_ID:?phase2 requires a Slurm array task id}"
            seed=$((PHASE2_SEED_BASE + task_index + 1))
            job_token="${SLURM_ARRAY_JOB_ID}_${task_index}"
            ;;
        *)
            die "Unknown worker phase: ${phase}"
            ;;
    esac

    run_dir="${EXPERIMENT_ROOT}/${phase}/seed_${seed}/job_${job_token}"
    mkdir -p "${run_dir}/logs"
    [[ ! -e "${run_dir}/COMPLETE" ]] || die "Refusing to overwrite completed run ${run_dir}."

    worker_exit() {
        exit_status=$?
        if (( exit_status == 0 )); then
            touch "${run_dir}/COMPLETE"
        else
            printf '%s\n' "${exit_status}" > "${run_dir}/FAILED"
        fi
    }
    trap worker_exit EXIT

    torch_threads=$((CPUS - NUM_WORKERS))
    export OMP_NUM_THREADS="${torch_threads}"
    export MKL_NUM_THREADS="${torch_threads}"
    export OPENBLAS_NUM_THREADS="${torch_threads}"
    export NUMEXPR_NUM_THREADS="${torch_threads}"
    export PYTHONUNBUFFERED=1

    {
        printf 'phase=%s\n' "${phase}"
        printf 'seed=%s\n' "${seed}"
        printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
        printf 'hostname=%s\n' "$(hostname)"
        printf 'git_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
        printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${run_dir}/run_info.txt"

    "${VENV_DIR}/bin/python" -m prost_t2_classification \
        --log-dir "${run_dir}/logs" \
        train \
        --manifest "${MANIFEST}" \
        --runs-dir "${run_dir}" \
        --mode both \
        --device cpu \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH_SIZE}" \
        --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --num-workers "${NUM_WORKERS}" \
        --patience "${PATIENCE}" \
        --lr "${LEARNING_RATE}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --seed "${seed}"

    "${VENV_DIR}/bin/python" - "${run_dir}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
real = list(root.glob("*_real/test_metrics.json"))
complex_runs = list(root.glob("*_complex_modrelu/test_metrics.json"))
if len(real) != 1 or len(complex_runs) != 1:
    raise SystemExit(
        f"Expected one completed real and complex run; found real={len(real)}, complex={len(complex_runs)}"
    )
PY
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${run_dir}/run_info.txt"
}

summarize_phase2() {
    local array_job="${1:?phase2 array job id is required}"
    "${VENV_DIR}/bin/python" - "${EXPERIMENT_ROOT}" "${array_job}" "${PHASE2_SEEDS}" "${PHASE2_SEED_BASE}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
array_job = sys.argv[2]
count = int(sys.argv[3])
seed_base = int(sys.argv[4])
records: list[dict[str, object]] = []

for index in range(count):
    seed = seed_base + index + 1
    job_dir = root / "phase2" / f"seed_{seed}" / f"job_{array_job}_{index}"
    if not (job_dir / "COMPLETE").is_file():
        raise SystemExit(f"Missing completion marker: {job_dir / 'COMPLETE'}")
    model_patterns = {"real": "*_real", "complex": "*_complex_modrelu"}
    for model, pattern in model_patterns.items():
        run_dirs = [path for path in job_dir.glob(pattern) if path.is_dir()]
        if len(run_dirs) != 1:
            raise SystemExit(f"Expected one {model} run in {job_dir}; found {len(run_dirs)}")
        run_dir = run_dirs[0]
        test = json.loads((run_dir / "test_metrics.json").read_text())
        threshold = json.loads((run_dir / "threshold.json").read_text())
        history = pd.read_csv(run_dir / "history.csv")
        best_row = history.loc[history["val_auc"].idxmax()]
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
metrics.to_csv(root / "metrics_by_seed.csv", index=False)

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
model_summary.columns = [f"{metric}_{stat}" for metric, stat in model_summary.columns]
model_summary.reset_index().to_csv(root / "summary_by_model.csv", index=False)

wide = metrics.pivot(index="seed", columns="model", values=available)
deltas = pd.DataFrame(index=wide.index)
for metric in available:
    deltas[f"{metric}_complex_minus_real"] = wide[(metric, "complex")] - wide[(metric, "real")]
deltas.reset_index().to_csv(root / "paired_deltas.csv", index=False)

rng = np.random.default_rng(73191)
paired_summary = []
for column in deltas.columns:
    values = deltas[column].to_numpy(dtype=float)
    bootstrap = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    paired_summary.append(
        {
            "metric": column.removesuffix("_complex_minus_real"),
            "n_pairs": len(values),
            "mean_complex_minus_real": values.mean(),
            "std_complex_minus_real": values.std(ddof=1),
            "ci95_low": np.quantile(bootstrap, 0.025),
            "ci95_high": np.quantile(bootstrap, 0.975),
            "complex_wins": int((values > 0).sum()),
            "ties": int((values == 0).sum()),
            "real_wins": int((values < 0).sum()),
        }
    )
pd.DataFrame(paired_summary).to_csv(root / "paired_summary.csv", index=False)
print(f"Wrote phase-two summaries for {count} paired seeds to {root}")
PY
    touch "${EXPERIMENT_ROOT}/PHASE2_COMPLETE"
}

case "${1:-submit}" in
    submit)
        submit_experiment
        ;;
    __worker)
        shift
        check_configuration
        run_worker "$@"
        ;;
    __summarize)
        shift
        summarize_phase2 "$@"
        ;;
    *)
        die "Usage: bash scripts/run_cluster_experiment.sh [submit]"
        ;;
esac
