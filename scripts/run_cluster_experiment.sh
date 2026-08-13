#!/usr/bin/env bash
set -Eeuo pipefail

# One-command paired real-vs-complex-pooling CPU/Slurm experiment.
#
# Expected repository layout after cloning branch "exp":
#   data/manifest.csv
#   data/samples/*.npz
#
# Run from anywhere inside the clone:
#   bash scripts/run_cluster_experiment.sh
#
# On CECI, load one recent Python module first (Python 3.10-3.12):
#   ml spider Python
#   ml load <the selected Python module>
#
# Optional overrides, for example:
#   PARTITION=general CPUS=20 MAX_PARALLEL=6 bash scripts/run_cluster_experiment.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
MANIFEST="${MANIFEST:-${DATA_DIR}/manifest.csv}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-pooling_vs_real_v1}"
PERSISTENT_RUNS_ROOT="${PERSISTENT_RUNS_ROOT:-${GLOBALSCRATCH:-${REPO_ROOT}/runs}/prost_t2_experiments}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PERSISTENT_RUNS_ROOT}/${EXPERIMENT_NAME}}"
CLUSTER_REQUIREMENTS="${CLUSTER_REQUIREMENTS:-${REPO_ROOT}/requirements-cluster.txt}"

PARTITION="${PARTITION:-work}"
CPUS="${CPUS:-16}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-1048:00:00}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"

PILOT_SEED="${PILOT_SEED:-10383}"
PHASE2_SEEDS="${PHASE2_SEEDS:-20}"
PHASE2_SEED_BASE="${PHASE2_SEED_BASE:-24000}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PATIENCE="${PATIENCE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
USE_SYSTEM_TORCH="${USE_SYSTEM_TORCH:-auto}"
STAGE_DATA="${STAGE_DATA:-1}"
SLURM_HINT="${SLURM_HINT:-nomultithread}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${GLOBALSCRATCH:-${HOME}}/.cache/pip-prost-t2}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-exp}"
COMPLEX_POOLINGS="max median average"
MODELS="real complex_max complex_median complex_average"
MODEL_COUNT=4
PRIMARY_ENDPOINT="test_auc"
PRIMARY_COMPARISON="complex_median_minus_real"

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
    [[ "${USE_SYSTEM_TORCH}" =~ ^(auto|0|1)$ ]] || die "USE_SYSTEM_TORCH must be auto, 0, or 1."
    [[ "${STAGE_DATA}" =~ ^[01]$ ]] || die "STAGE_DATA must be 0 or 1."
}

check_repository() {
    command -v git >/dev/null 2>&1 || die "git is not available."
    local branch
    branch="$(git -C "${REPO_ROOT}" branch --show-current)"
    [[ "${branch}" == "${EXPECTED_BRANCH}" ]] || die "Expected branch ${EXPECTED_BRANCH}; current branch is ${branch:-detached}."
    git -C "${REPO_ROOT}" diff --quiet || die "Tracked files have unstaged changes."
    git -C "${REPO_ROOT}" diff --cached --quiet || die "Tracked files have staged changes."
}

ensure_environment() {
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "${PYTHON_BIN} is not available. Set PYTHON_BIN to a cluster Python 3 executable."
    [[ -f "${CLUSTER_REQUIREMENTS}" ]] || die "Missing ${CLUSTER_REQUIREMENTS}."
    "${PYTHON_BIN}" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit(
        f"Python {sys.version.split()[0]} is unsupported by the pinned CPU environment; "
        "load a CECI Python 3.10, 3.11, or 3.12 module first"
    )
PY

    local system_torch=0 environment_mode
    if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import torch

version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
is_compatible_cpu_build = version >= (2, 2) and torch.version.cuda is None
raise SystemExit(0 if is_compatible_cpu_build else 1)
PY
    then
        system_torch=1
    fi
    if [[ "${USE_SYSTEM_TORCH}" == "1" && "${system_torch}" != "1" ]]; then
        die "USE_SYSTEM_TORCH=1, but the loaded Python module has no CPU-only PyTorch >=2.2."
    fi

    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        if [[ "${USE_SYSTEM_TORCH}" != "0" && "${system_torch}" == "1" ]]; then
            "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
            environment_mode="system-torch"
        else
            "${PYTHON_BIN}" -m venv "${VENV_DIR}"
            environment_mode="pinned-cpu-torch"
        fi
        printf '%s\n' "${environment_mode}" > "${VENV_DIR}/prost_t2_environment_mode"
    fi

    environment_mode="$(cat "${VENV_DIR}/prost_t2_environment_mode" 2>/dev/null || printf 'existing-venv')"
    export PIP_CACHE_DIR PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
    mkdir -p "${PIP_CACHE_DIR}"
    "${VENV_DIR}/bin/python" -m pip install \
        'pip==24.3.1' 'setuptools==75.6.0' 'wheel==0.45.1'
    "${VENV_DIR}/bin/python" -m pip install --requirement "${CLUSTER_REQUIREMENTS}"
    if [[ "${environment_mode}" != "system-torch" ]]; then
        "${VENV_DIR}/bin/python" -m pip install \
            "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
    fi
    "${VENV_DIR}/bin/python" -m pip install --no-deps --editable "${REPO_ROOT}"
    "${VENV_DIR}/bin/python" - <<'PY'
import platform
import sys

import numpy
import pandas
import sklearn
import torch
import tqdm
import prost_t2_classification

if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit(f"The virtual environment uses unsupported Python {platform.python_version()}")
if torch.version.cuda is not None:
    raise SystemExit(f"Expected a CPU-only PyTorch build, found CUDA {torch.version.cuda}")
print(
    f"Environment ready: Python {platform.python_version()}, PyTorch {torch.__version__}, "
    f"NumPy {numpy.__version__}, CPU-only"
)
PY
}

validate_data() {
    [[ -f "${MANIFEST}" ]] || die "Missing ${MANIFEST}. Copy manifest.csv and samples/ into ${DATA_DIR}."
    [[ -d "${DATA_DIR}/samples" ]] || die "Missing ${DATA_DIR}/samples."
    "${VENV_DIR}/bin/python" - "${MANIFEST}" <<'PY'
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from prost_t2_classification.dataset import validate_manifest

manifest_path = Path(sys.argv[1])
frame = pd.read_csv(manifest_path)
validate_manifest(frame)
missing_files = [value for value in frame["path"] if not (manifest_path.parent / str(value)).is_file()]
if missing_files:
    preview = ", ".join(map(str, missing_files[:5]))
    raise SystemExit(f"Manifest refers to {len(missing_files)} missing sample(s), including: {preview}")
split_counts = frame["data_split"].astype(str).str.strip().str.lower().value_counts().to_dict()
first_sample = manifest_path.parent / str(frame.iloc[0]["path"])
with np.load(first_sample) as sample:
    if "image_complex" not in sample:
        raise SystemExit(f"{first_sample} does not contain image_complex")
    image = sample["image_complex"]
if image.ndim != 3 or image.shape[0] != 4 or not np.iscomplexobj(image):
    raise SystemExit(f"{first_sample} image_complex must be a complex array with shape (4, height, width)")
print(f"Validated {len(frame)} samples; split counts: {split_counts}; channels: 4")
PY
}

export_worker_environment() {
    export REPO_ROOT DATA_DIR MANIFEST VENV_DIR EXPERIMENT_ROOT CPUS
    export PILOT_SEED PHASE2_SEEDS PHASE2_SEED_BASE
    export EPOCHS BATCH_SIZE GRADIENT_ACCUMULATION_STEPS NUM_WORKERS PATIENCE
    export LEARNING_RATE WEIGHT_DECAY STAGE_DATA COMPLEX_POOLINGS MODELS MODEL_COUNT
    export PRIMARY_ENDPOINT
    export PRIMARY_COMPARISON
}

record_loaded_modules() {
    if command -v module >/dev/null 2>&1; then
        module -t list 2>&1 || true
    elif [[ -n "${LOADEDMODULES:-}" ]]; then
        tr ':' '\n' <<< "${LOADEDMODULES}"
    else
        printf '%s\n' "No environment modules detected"
    fi
}

check_host() {
    check_configuration
    check_repository
    ensure_environment
    validate_data
    printf '\nHost check passed.\n'
    printf 'Host: %s\n' "$(hostname)"
    printf 'Python: %s\n' "$("${VENV_DIR}/bin/python" --version 2>&1)"
    printf 'Experiment root: %s\n' "${EXPERIMENT_ROOT}"
    printf 'Partition: %s; CPUs/job: %s; memory/job: %s\n' "${PARTITION}" "${CPUS}" "${MEMORY}"
    printf 'Stage NPZ data to job-local scratch: %s\n' "${STAGE_DATA}"
    printf 'Loaded modules:\n'
    record_loaded_modules
    command -v sbatch >/dev/null 2>&1 || die "Environment is valid, but sbatch is unavailable on this host."
}

write_submission_metadata() {
    local commit
    commit="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
    {
        printf 'git_commit=%s\n' "${commit}"
        printf 'manifest=%s\n' "${MANIFEST}"
        printf 'experiment_root=%s\n' "${EXPERIMENT_ROOT}"
        printf 'python_bin=%s\n' "$(command -v "${PYTHON_BIN}")"
        printf 'environment_mode=%s\n' "$(cat "${VENV_DIR}/prost_t2_environment_mode" 2>/dev/null || printf 'existing-venv')"
        printf 'torch_version_request=%s\n' "${TORCH_VERSION}"
        printf 'stage_data=%s\n' "${STAGE_DATA}"
        printf 'partition=%s\n' "${PARTITION}"
        printf 'cpus=%s\n' "${CPUS}"
        printf 'memory=%s\n' "${MEMORY}"
        printf 'time_limit=%s\n' "${TIME_LIMIT}"
        printf 'pilot_seed=%s\n' "${PILOT_SEED}"
        printf 'phase2_seeds=%s\n' "${PHASE2_SEEDS}"
        printf 'phase2_seed_base=%s\n' "${PHASE2_SEED_BASE}"
        printf 'phase2_seed_first=%s\n' "$((PHASE2_SEED_BASE + 1))"
        printf 'phase2_seed_last=%s\n' "$((PHASE2_SEED_BASE + PHASE2_SEEDS))"
        printf 'pilot_in_confirmatory_analysis=0\n'
        printf 'models=%s\n' "${MODELS// /,}"
        printf 'complex_poolings=%s\n' "${COMPLEX_POOLINGS// /,}"
        printf 'primary_endpoint=%s\n' "${PRIMARY_ENDPOINT}"
        printf 'primary_comparison=%s\n' "${PRIMARY_COMPARISON}"
        printf 'secondary_endpoints=test_average_precision,test_balanced_accuracy,test_sensitivity,test_specificity\n'
        printf 'epochs=%s\n' "${EPOCHS}"
        printf 'batch_size=%s\n' "${BATCH_SIZE}"
        printf 'gradient_accumulation_steps=%s\n' "${GRADIENT_ACCUMULATION_STEPS}"
        printf 'num_workers=%s\n' "${NUM_WORKERS}"
        printf 'patience=%s\n' "${PATIENCE}"
        printf 'learning_rate=%s\n' "${LEARNING_RATE}"
        printf 'weight_decay=%s\n' "${WEIGHT_DECAY}"
    } > "${EXPERIMENT_ROOT}/configuration.txt"
    "${VENV_DIR}/bin/python" -m pip freeze > "${EXPERIMENT_ROOT}/environment.txt"
    record_loaded_modules > "${EXPERIMENT_ROOT}/modules.txt"
}

submit_experiment() {
    command -v sbatch >/dev/null 2>&1 || die "sbatch is not available; run this command on a Slurm login node."
    check_configuration
    check_repository
    [[ ! -e "${EXPERIMENT_ROOT}/submission.txt" ]] || die "${EXPERIMENT_ROOT} was already submitted. Set a new EXPERIMENT_NAME."
    [[ ! -e "${EXPERIMENT_ROOT}/submission.partial" ]] || die "${EXPERIMENT_ROOT} has a partial submission. Inspect its job IDs before retrying."

    ensure_environment
    validate_data
    mkdir -p "${EXPERIMENT_ROOT}/slurm" "${EXPERIMENT_ROOT}/phase1" "${EXPERIMENT_ROOT}/phase2"
    write_submission_metadata
    export_worker_environment

    local pilot_job array_job summary_job pilot_spec submission_partial
    submission_partial="${EXPERIMENT_ROOT}/submission.partial"
    pilot_spec="0-$((MODEL_COUNT - 1))%${MODEL_COUNT}"
    pilot_job="$(sbatch --parsable \
        --job-name=prost-med-pilot \
        --partition="${PARTITION}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${CPUS}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --hint="${SLURM_HINT}" \
        --array="${pilot_spec}" \
        --chdir="${REPO_ROOT}" \
        --output="${EXPERIMENT_ROOT}/slurm/phase1_%A_%a.out" \
        --error="${EXPERIMENT_ROOT}/slurm/phase1_%A_%a.err" \
        --export=ALL \
        "${SCRIPT_PATH}" __worker phase1)"
    pilot_job="${pilot_job%%;*}"
    printf 'phase1_job=%s\n' "${pilot_job}" | tee "${submission_partial}"

    submit_phase2_jobs "${pilot_job}" "${submission_partial}"
    array_job="${SUBMITTED_PHASE2_ARRAY_JOB}"
    summary_job="${SUBMITTED_SUMMARY_JOB}"
    mv -- "${submission_partial}" "${EXPERIMENT_ROOT}/submission.txt"
    cat "${EXPERIMENT_ROOT}/submission.txt"

    printf '\nPhase two contains %s paired seeds x %s model tasks and starts only if all pilot models succeed.\n' \
        "${PHASE2_SEEDS}" "${MODEL_COUNT}"
    printf 'Monitor with: squeue -j %s,%s,%s\n' "${pilot_job}" "${array_job}" "${summary_job}"
    printf 'Results will be written under: %s\n' "${EXPERIMENT_ROOT}"
}

SUBMITTED_PHASE2_ARRAY_JOB=""
SUBMITTED_SUMMARY_JOB=""

submit_phase2_jobs() {
    local dependency_job="${1:-}"
    local submission_file="${2:?submission metadata file is required}"
    local array_job summary_job array_last array_spec
    local -a dependency_args=()
    if [[ -n "${dependency_job}" ]]; then
        dependency_args=(--dependency="afterok:${dependency_job}")
    fi

    array_last=$((PHASE2_SEEDS * MODEL_COUNT - 1))
    array_spec="0-${array_last}%${MAX_PARALLEL}"
    array_job="$(sbatch --parsable \
        --job-name=prost-med-seeds \
        --partition="${PARTITION}" \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task="${CPUS}" \
        --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" \
        --hint="${SLURM_HINT}" \
        --array="${array_spec}" \
        "${dependency_args[@]}" \
        --chdir="${REPO_ROOT}" \
        --output="${EXPERIMENT_ROOT}/slurm/phase2_%A_%a.out" \
        --error="${EXPERIMENT_ROOT}/slurm/phase2_%A_%a.err" \
        --export=ALL \
        "${SCRIPT_PATH}" __worker phase2)"
    array_job="${array_job%%;*}"
    printf 'phase2_array_job=%s\n' "${array_job}" | tee -a "${submission_file}"

    summary_job="$(sbatch --parsable \
        --job-name=prost-med-summary \
        --partition="${PARTITION}" \
        --nodes=1 \
        --ntasks=1 \
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
    printf 'summary_job=%s\n' "${summary_job}" | tee -a "${submission_file}"

    SUBMITTED_PHASE2_ARRAY_JOB="${array_job}"
    SUBMITTED_SUMMARY_JOB="${summary_job}"
}

submission_value() {
    local file="${1:?submission file is required}"
    local key="${2:?submission key is required}"
    awk -F= -v key="${key}" '$1 == key { print $2; exit }' "${file}"
}

validate_resume_configuration() {
    local configuration="${EXPERIMENT_ROOT}/configuration.txt"
    [[ -f "${configuration}" ]] || die "Missing original configuration: ${configuration}"

    local key expected actual
    while IFS='|' read -r key expected; do
        actual="$(submission_value "${configuration}" "${key}")"
        [[ -n "${actual}" ]] || die "Original configuration is missing ${key}."
        [[ "${actual}" == "${expected}" ]] || \
            die "Cannot resume with ${key}=${expected}; original submission used ${actual}."
    done <<EOF
pilot_seed|${PILOT_SEED}
phase2_seeds|${PHASE2_SEEDS}
phase2_seed_base|${PHASE2_SEED_BASE}
models|${MODELS// /,}
epochs|${EPOCHS}
batch_size|${BATCH_SIZE}
gradient_accumulation_steps|${GRADIENT_ACCUMULATION_STEPS}
num_workers|${NUM_WORKERS}
patience|${PATIENCE}
learning_rate|${LEARNING_RATE}
weight_decay|${WEIGHT_DECAY}
EOF
}

validate_completed_pilots() {
    "${VENV_DIR}/bin/python" - "${EXPERIMENT_ROOT}" "${PILOT_SEED}" <<'PY'
import csv
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]) / "phase1" / f"seed_{sys.argv[2]}"
patterns = {
    "real": "*_real",
    "complex_max": "*_complex_modrelu",
    "complex_median": "*_complex_modrelu_median_pool",
    "complex_average": "*_complex_modrelu_average_pool",
}
required = ("config.json", "history.csv", "threshold.json", "test_metrics.json")
for model, pattern in patterns.items():
    matches = [path for path in root.glob(f"job_*_0/{pattern}") if path.is_dir()]
    if len(matches) != 1:
        raise SystemExit(
            f"Expected one completed pilot run for {model} under {root}; found {len(matches)}"
        )
    run = matches[0]
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise SystemExit(f"Incomplete pilot {model} in {run}: missing {', '.join(missing)}")
    with (run / "history.csv").open(newline="", encoding="utf-8") as handle:
        if not any(csv.DictReader(handle)):
            raise SystemExit(f"Pilot history is empty: {run / 'history.csv'}")
    with (run / "test_metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    for metric in ("auc", "average_precision", "balanced_accuracy"):
        if metric not in metrics:
            raise SystemExit(f"Pilot {model} is missing test metric {metric}")
    print(f"Validated completed pilot {model}: {run}")
PY
}

resume_phase2() {
    command -v sbatch >/dev/null 2>&1 || die "sbatch is not available; run this command on a Slurm login node."
    command -v squeue >/dev/null 2>&1 || die "squeue is not available; run this command on a Slurm login node."
    check_configuration
    check_repository

    local original_submission="${EXPERIMENT_ROOT}/submission.txt"
    local recovery_file="${EXPERIMENT_ROOT}/phase2_resume.txt"
    local recovery_partial="${EXPERIMENT_ROOT}/phase2_resume.partial"
    [[ -f "${original_submission}" ]] || die "Missing original submission: ${original_submission}"
    [[ ! -e "${recovery_file}" ]] || die "Phase two was already resumed; inspect ${recovery_file}."
    [[ ! -e "${recovery_partial}" ]] || die "A partial phase-two recovery exists: ${recovery_partial}"
    [[ ! -e "${EXPERIMENT_ROOT}/PHASE2_COMPLETE" ]] || die "Phase two is already complete."
    if find "${EXPERIMENT_ROOT}/phase2" -name test_metrics.json -print -quit | grep -q .; then
        die "Phase two already contains test results; refusing an automatic recovery."
    fi

    local obsolete_array obsolete_summary queued
    obsolete_array="$(submission_value "${original_submission}" phase2_array_job)"
    obsolete_summary="$(submission_value "${original_submission}" summary_job)"
    [[ -n "${obsolete_array}" && -n "${obsolete_summary}" ]] || \
        die "Original submission does not contain phase-two and summary job IDs."
    queued="$(squeue -h -j "${obsolete_array},${obsolete_summary}" 2>/dev/null || true)"
    [[ -z "${queued}" ]] || \
        die "Obsolete jobs are still queued. Run: scancel ${obsolete_array} ${obsolete_summary}"

    ensure_environment
    validate_data
    validate_resume_configuration
    validate_completed_pilots
    export_worker_environment

    {
        printf 'recovery_git_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
        printf 'resumed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'superseded_phase2_array_job=%s\n' "${obsolete_array}"
        printf 'superseded_summary_job=%s\n' "${obsolete_summary}"
    } > "${recovery_partial}"
    submit_phase2_jobs "" "${recovery_partial}"
    mv -- "${recovery_partial}" "${recovery_file}"
    cat "${recovery_file}"
    printf '\nValidated the four completed pilots; they will not be rerun.\n'
    printf 'Monitor with: squeue -j %s,%s\n' \
        "${SUBMITTED_PHASE2_ARRAY_JOB}" "${SUBMITTED_SUMMARY_JOB}"
    printf 'Results will be written under: %s\n' "${EXPERIMENT_ROOT}"
}

stage_worker_data() {
    if [[ "${STAGE_DATA}" == "0" ]]; then
        printf '%s\n' "${MANIFEST}"
        return
    fi

    local scratch_root="${LOCALSCRATCH:-${TMPDIR:-}}"
    [[ -n "${scratch_root}" && -d "${scratch_root}" ]] || \
        die "STAGE_DATA=1, but neither LOCALSCRATCH nor TMPDIR points to a directory."
    local staged_data="${scratch_root}/prost_t2_data_${SLURM_JOB_ID}"
    mkdir -p "${staged_data}"
    cp -a "${DATA_DIR}/." "${staged_data}/"
    [[ -f "${staged_data}/manifest.csv" ]] || die "Data staging did not produce manifest.csv."
    printf '%s\n' "${staged_data}/manifest.csv"
}

run_worker() {
    local phase="${1:?worker phase is required}"
    local seed task_index seed_index model_index model_key mode pooling completion_marker failure_marker
    local job_token run_dir torch_threads worker_manifest
    local run_dir_q completion_marker_q failure_marker_q
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "The worker must run inside a Slurm allocation."
    [[ -x "${VENV_DIR}/bin/python" ]] || die "Missing experiment environment at ${VENV_DIR}."

    case "${phase}" in
        phase1)
            seed="${PILOT_SEED}"
            task_index="${SLURM_ARRAY_TASK_ID:?phase1 requires a Slurm array task id}"
            seed_index=0
            model_index="${task_index}"
            job_token="${SLURM_ARRAY_JOB_ID}_0"
            ;;
        phase2)
            task_index="${SLURM_ARRAY_TASK_ID:?phase2 requires a Slurm array task id}"
            seed_index=$((task_index / MODEL_COUNT))
            model_index=$((task_index % MODEL_COUNT))
            seed=$((PHASE2_SEED_BASE + seed_index + 1))
            job_token="${SLURM_ARRAY_JOB_ID}_${seed_index}"
            ;;
        *)
            die "Unknown worker phase: ${phase}"
            ;;
    esac

    case "${model_index}" in
        0) model_key="real"; mode="real"; pooling="none" ;;
        1) model_key="complex_max"; mode="complex"; pooling="max" ;;
        2) model_key="complex_median"; mode="complex"; pooling="median" ;;
        3) model_key="complex_average"; mode="complex"; pooling="average" ;;
        *) die "Unknown model index: ${model_index}" ;;
    esac
    completion_marker="${model_key^^}_COMPLETE"
    failure_marker="${model_key^^}_FAILED"

    run_dir="${EXPERIMENT_ROOT}/${phase}/seed_${seed}/job_${job_token}"
    mkdir -p "${run_dir}/logs/${model_key}"
    [[ ! -e "${run_dir}/${completion_marker}" ]] || \
        die "Refusing to overwrite completed ${model_key} run in ${run_dir}."

    worker_exit() {
        local exit_status=$?
        local worker_run_dir="${1:?worker run directory is required}"
        local worker_completion_marker="${2:?worker completion marker is required}"
        local worker_failure_marker="${3:?worker failure marker is required}"
        if (( exit_status == 0 )); then
            rm -f -- "${worker_run_dir}/${worker_failure_marker}"
            touch "${worker_run_dir}/${worker_completion_marker}"
        else
            printf '%s\n' "${exit_status}" > "${worker_run_dir}/${worker_failure_marker}"
        fi
    }
    printf -v run_dir_q '%q' "${run_dir}"
    printf -v completion_marker_q '%q' "${completion_marker}"
    printf -v failure_marker_q '%q' "${failure_marker}"
    trap "worker_exit ${run_dir_q} ${completion_marker_q} ${failure_marker_q}" EXIT

    torch_threads=$((CPUS - NUM_WORKERS))
    export OMP_NUM_THREADS="${torch_threads}"
    export MKL_NUM_THREADS="${torch_threads}"
    export OPENBLAS_NUM_THREADS="${torch_threads}"
    export NUMEXPR_NUM_THREADS="${torch_threads}"
    export OMP_DYNAMIC=FALSE
    export OMP_PROC_BIND=close
    export OMP_PLACES=cores
    export MALLOC_ARENA_MAX=4
    export PYTHONUNBUFFERED=1

    worker_manifest="$(stage_worker_data)"

    {
        printf 'phase=%s\n' "${phase}"
        printf 'seed=%s\n' "${seed}"
        printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID}"
        printf 'hostname=%s\n' "$(hostname)"
        printf 'source_manifest=%s\n' "${MANIFEST}"
        printf 'worker_manifest=%s\n' "${worker_manifest}"
        printf 'python=%s\n' "${VENV_DIR}/bin/python"
        printf 'omp_num_threads=%s\n' "${OMP_NUM_THREADS}"
        printf 'git_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
        printf 'model=%s\n' "${model_key}"
        printf 'mode=%s\n' "${mode}"
        printf 'pooling=%s\n' "${pooling}"
        printf 'models=%s\n' "${MODELS// /,}"
        printf 'complex_poolings=%s\n' "${COMPLEX_POOLINGS// /,}"
        printf 'primary_comparison=%s\n' "${PRIMARY_COMPARISON}"
        printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${run_dir}/run_info_${model_key}.txt"
    uname -a > "${run_dir}/host_${model_key}.txt"
    command -v lscpu >/dev/null 2>&1 && lscpu >> "${run_dir}/host_${model_key}.txt"
    record_loaded_modules > "${run_dir}/modules_${model_key}.txt"
    "${VENV_DIR}/bin/python" - <<'PY' > "${run_dir}/torch_runtime_${model_key}.txt"
import torch

print(f"torch={torch.__version__}")
print(f"num_threads={torch.get_num_threads()}")
print(f"num_interop_threads={torch.get_num_interop_threads()}")
print(torch.__config__.show())
PY

    local -a train_args
    train_args=(
        --log-dir "${run_dir}/logs/${model_key}"
        train
        --manifest "${worker_manifest}"
        --runs-dir "${run_dir}"
        --device cpu
        --epochs "${EPOCHS}"
        --batch-size "${BATCH_SIZE}"
        --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
        --num-workers "${NUM_WORKERS}"
        --patience "${PATIENCE}"
        --lr "${LEARNING_RATE}"
        --weight-decay "${WEIGHT_DECAY}"
        --seed "${seed}"
    )

    if [[ "${mode}" == "real" ]]; then
        "${VENV_DIR}/bin/python" -m prost_t2_classification \
            "${train_args[@]}" --mode real
    else
        "${VENV_DIR}/bin/python" -m prost_t2_classification \
            "${train_args[@]}" --mode complex --complex-pooling "${pooling}"
    fi

    "${VENV_DIR}/bin/python" - "${run_dir}" "${model_key}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
model = sys.argv[2]
patterns = {
    "real": "*_real/test_metrics.json",
    "complex_max": "*_complex_modrelu/test_metrics.json",
    "complex_median": "*_complex_modrelu_median_pool/test_metrics.json",
    "complex_average": "*_complex_modrelu_average_pool/test_metrics.json",
}
count = len(list(root.glob(patterns[model])))
if count != 1:
    raise SystemExit(f"Expected one completed {model} run; found {count}")
PY
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> \
        "${run_dir}/run_info_${model_key}.txt"
}

summarize_phase2() {
    local array_job="${1:?phase2 array job id is required}"
    "${VENV_DIR}/bin/python" -m prost_t2_classification.experiment_summary \
        --experiment-root "${EXPERIMENT_ROOT}" \
        --array-job "${array_job}" \
        --count "${PHASE2_SEEDS}" \
        --seed-base "${PHASE2_SEED_BASE}"
    touch "${EXPERIMENT_ROOT}/PHASE2_COMPLETE"
}

case "${1:-submit}" in
    submit)
        submit_experiment
        ;;
    check)
        check_host
        ;;
    resume-phase2)
        resume_phase2
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
        die "Usage: bash scripts/run_cluster_experiment.sh [check|submit|resume-phase2]"
        ;;
esac
