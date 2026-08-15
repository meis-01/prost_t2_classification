#!/usr/bin/env bash
set -Eeuo pipefail

# Full-factorial real-vs-complex CPU/Slurm experiment.
# Run on the exp branch with:
#   bash scripts/run_cluster_experiment.sh check
#   bash scripts/run_cluster_experiment.sh submit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
MANIFEST="${MANIFEST:-${DATA_DIR}/manifest.csv}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-complex_full_factorial_v1}"
PERSISTENT_RUNS_ROOT="${PERSISTENT_RUNS_ROOT:-${GLOBALSCRATCH:-${REPO_ROOT}/runs}/prost_t2_experiments}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PERSISTENT_RUNS_ROOT}/${EXPERIMENT_NAME}}"
CLUSTER_REQUIREMENTS="${CLUSTER_REQUIREMENTS:-${REPO_ROOT}/requirements-cluster.txt}"

PARTITION="${PARTITION:-work}"
CPUS="${CPUS:-16}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-48:00:00}"
# Zero means no launcher-side throttle; Slurm/QoS supplies the real limit.
MAX_PARALLEL="${MAX_PARALLEL:-0}"

PILOT_SEED="${PILOT_SEED:-10383}"
PHASE2_SEEDS="${PHASE2_SEEDS:-4}"
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

MODEL_COUNT=481
COMPLEX_MODEL_COUNT=480
PRIMARY_ENDPOINT="test_auc"
PRIMARY_COMPARISON="complex_median_minus_real"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

require_positive_integer() {
    local name="$1" value="$2"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer; got ${value}."
}

check_configuration() {
    require_positive_integer CPUS "${CPUS}"
    [[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]] || die "MAX_PARALLEL must be zero or a positive integer."
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

model_array_spec() {
    local last=$((MODEL_COUNT - 1))
    if (( MAX_PARALLEL == 0 )); then
        printf '0-%s\n' "${last}"
    else
        printf '0-%s%%%s\n' "${last}" "${MAX_PARALLEL}"
    fi
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
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "${PYTHON_BIN} is unavailable. Load a cluster Python 3 module."
    [[ -f "${CLUSTER_REQUIREMENTS}" ]] || die "Missing ${CLUSTER_REQUIREMENTS}."
    "${PYTHON_BIN}" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit(f"Python {sys.version.split()[0]} is unsupported; load Python 3.10-3.12")
PY

    local system_torch=0 environment_mode
    if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import torch
version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
raise SystemExit(0 if version >= (2, 2) and torch.version.cuda is None else 1)
PY
    then
        system_torch=1
    fi
    if [[ "${USE_SYSTEM_TORCH}" == "1" && "${system_torch}" != "1" ]]; then
        die "USE_SYSTEM_TORCH=1 but the loaded Python has no compatible CPU-only PyTorch."
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
    "${VENV_DIR}/bin/python" -m pip install 'pip==24.3.1' 'setuptools==75.6.0' 'wheel==0.45.1'
    "${VENV_DIR}/bin/python" -m pip install --requirement "${CLUSTER_REQUIREMENTS}"
    if [[ "${environment_mode}" != "system-torch" ]]; then
        "${VENV_DIR}/bin/python" -m pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
    fi
    "${VENV_DIR}/bin/python" -m pip install --no-deps --editable "${REPO_ROOT}"
    "${VENV_DIR}/bin/python" - <<'PY'
import platform, torch
from prost_t2_classification.experiment_grid import experiment_model_specs
if torch.version.cuda is not None:
    raise SystemExit(f"Expected CPU-only PyTorch, found CUDA {torch.version.cuda}")
if len(experiment_model_specs()) != 481:
    raise SystemExit("Experiment grid does not contain exactly 481 models")
print(f"Environment ready: Python {platform.python_version()}, PyTorch {torch.__version__}, 481 models")
PY
}

validate_data() {
    [[ -f "${MANIFEST}" ]] || die "Missing ${MANIFEST}."
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
missing = [value for value in frame["path"] if not (manifest_path.parent / str(value)).is_file()]
if missing:
    raise SystemExit(f"Manifest refers to {len(missing)} missing samples")
first = manifest_path.parent / str(frame.iloc[0]["path"])
with np.load(first) as sample:
    image = sample.get("image_complex")
if image is None or image.ndim != 3 or image.shape[0] != 4 or not np.iscomplexobj(image):
    raise SystemExit(f"{first} must contain a complex image_complex array shaped (4, H, W)")
print(f"Validated {len(frame)} samples and all manifest paths")
PY
}

check_slurm_array_capacity() {
    command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable on this host."
    if command -v scontrol >/dev/null 2>&1; then
        local max_array_size
        max_array_size="$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/ {gsub(/[[:space:]]/, "", $2); print $2; exit}')"
        if [[ "${max_array_size}" =~ ^[1-9][0-9]*$ ]]; then
            (( MODEL_COUNT <= max_array_size )) || die "Pilot needs ${MODEL_COUNT} array entries; cluster MaxArraySize is ${max_array_size}."
        fi
    fi
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

export_worker_environment() {
    export REPO_ROOT DATA_DIR MANIFEST VENV_DIR EXPERIMENT_ROOT CPUS
    export PILOT_SEED PHASE2_SEEDS PHASE2_SEED_BASE MODEL_COUNT
    export EPOCHS BATCH_SIZE GRADIENT_ACCUMULATION_STEPS NUM_WORKERS PATIENCE
    export LEARNING_RATE WEIGHT_DECAY STAGE_DATA PRIMARY_ENDPOINT PRIMARY_COMPARISON
}

write_submission_metadata() {
    "${VENV_DIR}/bin/python" -m prost_t2_classification.experiment_grid write \
        --csv "${EXPERIMENT_ROOT}/model_grid.csv" \
        --json "${EXPERIMENT_ROOT}/model_grid.json"
    {
        printf 'git_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
        printf 'manifest=%s\n' "${MANIFEST}"
        printf 'experiment_root=%s\n' "${EXPERIMENT_ROOT}"
        printf 'model_count=%s\n' "${MODEL_COUNT}"
        printf 'complex_model_count=%s\n' "${COMPLEX_MODEL_COUNT}"
        printf 'grid=domain:2,pooling:3,normalization:2,convolution:2,stream_interaction:5,activation:4\n'
        printf 'pilot_seed=%s\n' "${PILOT_SEED}"
        printf 'phase2_seeds=%s\n' "${PHASE2_SEEDS}"
        printf 'phase2_seed_base=%s\n' "${PHASE2_SEED_BASE}"
        printf 'phase2_task_count=%s\n' "$((PHASE2_SEEDS * MODEL_COUNT))"
        printf 'phase2_array_count=%s\n' "${PHASE2_SEEDS}"
        printf 'phase2_execution_order=one_complete_model_array_per_seed\n'
        printf 'pilot_to_phase2_dependency=afterany\n'
        printf 'phase2_inter_seed_dependency=afterany\n'
        printf 'pilot_in_confirmatory_analysis=0\n'
        printf 'primary_endpoint=%s\n' "${PRIMARY_ENDPOINT}"
        printf 'primary_comparison=%s\n' "${PRIMARY_COMPARISON}"
        printf 'partition=%s\ncpus=%s\nmemory=%s\ntime_limit=%s\n' "${PARTITION}" "${CPUS}" "${MEMORY}" "${TIME_LIMIT}"
        printf 'max_parallel=%s\nepochs=%s\nbatch_size=%s\ngradient_accumulation_steps=%s\n' \
            "${MAX_PARALLEL}" "${EPOCHS}" "${BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}"
        printf 'launcher_parallel_cap=%s\n' "$([[ "${MAX_PARALLEL}" == "0" ]] && printf 'none' || printf '%s' "${MAX_PARALLEL}")"
        printf 'num_workers=%s\npatience=%s\nlearning_rate=%s\nweight_decay=%s\n' \
            "${NUM_WORKERS}" "${PATIENCE}" "${LEARNING_RATE}" "${WEIGHT_DECAY}"
        printf 'stage_data=%s\ntorch_version_request=%s\n' "${STAGE_DATA}" "${TORCH_VERSION}"
    } > "${EXPERIMENT_ROOT}/configuration.txt"
    "${VENV_DIR}/bin/python" -m pip freeze > "${EXPERIMENT_ROOT}/environment.txt"
    record_loaded_modules > "${EXPERIMENT_ROOT}/modules.txt"
}

check_host() {
    check_configuration
    check_repository
    ensure_environment
    validate_data
    check_slurm_array_capacity
    printf '\nHost check passed: 481 pilot tasks; %s confirmatory tasks in %s sequential seed arrays.\n' \
        "$((PHASE2_SEEDS * MODEL_COUNT))" "${PHASE2_SEEDS}"
    if (( MAX_PARALLEL == 0 )); then
        printf 'Launcher concurrency cap: none; Slurm/QoS controls active task count.\n'
    else
        printf 'Launcher concurrency cap: %s tasks.\n' "${MAX_PARALLEL}"
    fi
}

SUBMITTED_PHASE2_ARRAY_JOBS=""
SUBMITTED_SUMMARY_JOB=""

submit_phase2_jobs() {
    local dependency_job="${1:-}" submission_file="${2:?submission file is required}"
    local seed_index seed array_spec
    array_spec="$(model_array_spec)"
    local array_job summary_job previous_job="${dependency_job}"
    local -a dependency_args=()
    SUBMITTED_PHASE2_ARRAY_JOBS=""
    for ((seed_index = 0; seed_index < PHASE2_SEEDS; seed_index++)); do
        seed=$((PHASE2_SEED_BASE + seed_index + 1))
        dependency_args=()
        if [[ -n "${previous_job}" ]]; then
            dependency_args=(--dependency="afterany:${previous_job}")
        fi
        array_job="$(sbatch --parsable \
            --job-name=prost-grid-s${seed} --partition="${PARTITION}" \
            --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" --mem="${MEMORY}" \
            --time="${TIME_LIMIT}" --hint="${SLURM_HINT}" --array="${array_spec}" \
            "${dependency_args[@]}" --chdir="${REPO_ROOT}" \
            --output="${EXPERIMENT_ROOT}/slurm/phase2_seed${seed}_%A_%a.out" \
            --error="${EXPERIMENT_ROOT}/slurm/phase2_seed${seed}_%A_%a.err" \
            --export="ALL,PHASE2_SEED_INDEX=${seed_index}" "${SCRIPT_PATH}" __worker phase2)"
        array_job="${array_job%%;*}"
        printf 'phase2_seed_%s_job=%s\n' "${seed}" "${array_job}" | tee -a "${submission_file}"
        SUBMITTED_PHASE2_ARRAY_JOBS="${SUBMITTED_PHASE2_ARRAY_JOBS:+${SUBMITTED_PHASE2_ARRAY_JOBS},}${array_job}"
        previous_job="${array_job}"
    done
    printf 'phase2_array_jobs=%s\n' "${SUBMITTED_PHASE2_ARRAY_JOBS}" | tee -a "${submission_file}"
    summary_job="$(sbatch --parsable \
        --job-name=prost-grid-summary --partition="${PARTITION}" --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=8G --time=02:00:00 --dependency="afterany:${previous_job}" \
        --chdir="${REPO_ROOT}" --output="${EXPERIMENT_ROOT}/slurm/summary_%j.out" \
        --error="${EXPERIMENT_ROOT}/slurm/summary_%j.err" --export=ALL \
        "${SCRIPT_PATH}" __summarize)"
    SUBMITTED_SUMMARY_JOB="${summary_job%%;*}"
    printf 'summary_job=%s\n' "${SUBMITTED_SUMMARY_JOB}" | tee -a "${submission_file}"
}

submit_experiment() {
    check_configuration
    check_repository
    check_slurm_array_capacity
    [[ ! -e "${EXPERIMENT_ROOT}/submission.txt" ]] || die "Experiment was already submitted; choose a new EXPERIMENT_NAME."
    [[ ! -e "${EXPERIMENT_ROOT}/submission.partial" ]] || die "Partial submission exists; inspect it before retrying."
    ensure_environment
    validate_data
    mkdir -p "${EXPERIMENT_ROOT}/slurm" "${EXPERIMENT_ROOT}/phase1" "${EXPERIMENT_ROOT}/phase2"
    write_submission_metadata
    export_worker_environment

    local pilot_job pilot_spec
    pilot_spec="$(model_array_spec)"
    local submission_partial="${EXPERIMENT_ROOT}/submission.partial"
    pilot_job="$(sbatch --parsable \
        --job-name=prost-grid-pilot --partition="${PARTITION}" --nodes=1 --ntasks=1 \
        --cpus-per-task="${CPUS}" --mem="${MEMORY}" --time="${TIME_LIMIT}" \
        --hint="${SLURM_HINT}" --array="${pilot_spec}" --chdir="${REPO_ROOT}" \
        --output="${EXPERIMENT_ROOT}/slurm/phase1_%A_%a.out" \
        --error="${EXPERIMENT_ROOT}/slurm/phase1_%A_%a.err" --export=ALL \
        "${SCRIPT_PATH}" __worker phase1)"
    pilot_job="${pilot_job%%;*}"
    printf 'phase1_job=%s\n' "${pilot_job}" | tee "${submission_partial}"
    submit_phase2_jobs "${pilot_job}" "${submission_partial}"
    mv -- "${submission_partial}" "${EXPERIMENT_ROOT}/submission.txt"
    cat "${EXPERIMENT_ROOT}/submission.txt"
    printf '\nPilot: %s models. Phase two: %s seeds x %s models = %s tasks.\n' \
        "${MODEL_COUNT}" "${PHASE2_SEEDS}" "${MODEL_COUNT}" "$((PHASE2_SEEDS * MODEL_COUNT))"
    printf 'Monitor: squeue -j %s,%s,%s\n' "${pilot_job}" "${SUBMITTED_PHASE2_ARRAY_JOBS}" "${SUBMITTED_SUMMARY_JOB}"
}

submission_value() {
    awk -F= -v key="${2:?key is required}" '$1 == key { print $2; exit }' "${1:?file is required}"
}

validate_resume_configuration() {
    local configuration="${EXPERIMENT_ROOT}/configuration.txt" key expected actual
    [[ -f "${configuration}" ]] || die "Missing original configuration."
    while IFS='|' read -r key expected; do
        actual="$(submission_value "${configuration}" "${key}")"
        [[ "${actual}" == "${expected}" ]] || die "Resume mismatch for ${key}: current=${expected}, original=${actual}."
    done <<EOF
model_count|${MODEL_COUNT}
pilot_seed|${PILOT_SEED}
phase2_seeds|${PHASE2_SEEDS}
phase2_seed_base|${PHASE2_SEED_BASE}
epochs|${EPOCHS}
batch_size|${BATCH_SIZE}
gradient_accumulation_steps|${GRADIENT_ACCUMULATION_STEPS}
num_workers|${NUM_WORKERS}
patience|${PATIENCE}
learning_rate|${LEARNING_RATE}
weight_decay|${WEIGHT_DECAY}
EOF
}

resume_phase2() {
    check_configuration
    check_repository
    check_slurm_array_capacity
    command -v squeue >/dev/null 2>&1 || die "squeue is unavailable."
    local original="${EXPERIMENT_ROOT}/submission.txt" recovery="${EXPERIMENT_ROOT}/phase2_resume.txt"
    local partial="${EXPERIMENT_ROOT}/phase2_resume.partial" queued_ids summary_job queued
    [[ -f "${original}" ]] || die "Missing original submission."
    [[ ! -e "${recovery}" && ! -e "${partial}" ]] || die "A phase-two recovery already exists."
    [[ ! -e "${EXPERIMENT_ROOT}/PHASE2_COMPLETE" ]] || die "Phase two is already complete."
    queued_ids="$(submission_value "${original}" phase2_array_jobs)"
    summary_job="$(submission_value "${original}" summary_job)"
    queued="$(squeue -h -j "${queued_ids},${summary_job}" 2>/dev/null || true)"
    [[ -z "${queued}" ]] || die "Original phase-two jobs are still queued; cancel them before recovery."
    ensure_environment
    validate_data
    validate_resume_configuration
    export_worker_environment
    {
        printf 'recovery_git_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
        printf 'resumed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'superseded_phase2_array_jobs=%s\n' "${queued_ids}"
        printf 'superseded_summary_job=%s\n' "${summary_job}"
    } > "${partial}"
    submit_phase2_jobs "" "${partial}"
    mv -- "${partial}" "${recovery}"
    printf 'Recovery submitted. Completed model/seed directories will be skipped.\n'
}

stage_worker_data() {
    if [[ "${STAGE_DATA}" == "0" ]]; then
        printf '%s\n' "${MANIFEST}"
        return
    fi
    local scratch_root="${LOCALSCRATCH:-${TMPDIR:-}}"
    [[ -n "${scratch_root}" && -d "${scratch_root}" ]] || die "STAGE_DATA=1 but no LOCALSCRATCH or TMPDIR exists."
    local staged_data="${scratch_root}/prost_t2_data_${SLURM_JOB_ID}"
    mkdir -p "${staged_data}"
    cp -a "${DATA_DIR}/." "${staged_data}/"
    [[ -f "${staged_data}/manifest.csv" ]] || die "Data staging did not produce manifest.csv."
    printf '%s\n' "${staged_data}/manifest.csv"
}

run_worker() {
    local phase="${1:?worker phase is required}" local_task global_task seed_index seed model_index
    local model_key mode input_domain pooling normalization convolution streams interaction activation
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "Worker must run inside Slurm."
    [[ -x "${VENV_DIR}/bin/python" ]] || die "Missing experiment virtual environment."
    local_task="${SLURM_ARRAY_TASK_ID:?array task id is required}"
    if [[ "${phase}" == "phase1" ]]; then
        global_task="${local_task}"
        seed="${PILOT_SEED}"
        model_index="${global_task}"
    elif [[ "${phase}" == "phase2" ]]; then
        seed_index="${PHASE2_SEED_INDEX:?phase2 requires PHASE2_SEED_INDEX}"
        (( seed_index < PHASE2_SEEDS )) || die "Computed seed index is outside phase two."
        seed=$((PHASE2_SEED_BASE + seed_index + 1))
        model_index="${local_task}"
    else
        die "Unknown worker phase: ${phase}"
    fi
    IFS=$'\t' read -r model_key mode input_domain pooling normalization convolution streams interaction activation < <(
        "${VENV_DIR}/bin/python" -m prost_t2_classification.experiment_grid row --index "${model_index}"
    )
    [[ -n "${model_key}" ]] || die "Could not resolve model index ${model_index}."

    local model_dir="${EXPERIMENT_ROOT}/${phase}/seed_${seed}/models/${model_key}"
    if [[ -f "${model_dir}/COMPLETE" ]]; then
        printf 'Skipping completed %s seed %s.\n' "${model_key}" "${seed}"
        return
    fi
    local attempt_dir="${model_dir}/attempt_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    mkdir -p "${attempt_dir}/logs"
    local failure_marker="${attempt_dir}/FAILED" success_marker="${attempt_dir}/SUCCESS"
    local completion_marker="${model_dir}/COMPLETE"
    worker_exit() {
        local exit_status=$?
        trap - EXIT
        if (( exit_status == 0 )); then
            if ! rm -f -- "${failure_marker}" || ! touch "${success_marker}" || ! touch "${completion_marker}"; then
                exit_status=1
                rm -f -- "${success_marker}" "${completion_marker}" || true
                printf '%s\n' "Could not write success markers" > "${failure_marker}" || true
            fi
        else
            printf '%s\n' "${exit_status}" > "${failure_marker}" || true
        fi
        exit "${exit_status}"
    }
    trap worker_exit EXIT

    local torch_threads=$((CPUS - NUM_WORKERS)) worker_manifest
    export OMP_NUM_THREADS="${torch_threads}" MKL_NUM_THREADS="${torch_threads}"
    export OPENBLAS_NUM_THREADS="${torch_threads}" NUMEXPR_NUM_THREADS="${torch_threads}"
    export OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores MALLOC_ARENA_MAX=4 PYTHONUNBUFFERED=1
    worker_manifest="$(stage_worker_data)"
    {
        printf 'phase=%s\nseed=%s\nmodel_index=%s\nmodel_key=%s\n' "${phase}" "${seed}" "${model_index}" "${model_key}"
        printf 'mode=%s\ninput_domain=%s\npooling=%s\nnormalization=%s\n' "${mode}" "${input_domain}" "${pooling}" "${normalization}"
        printf 'convolution=%s\nstreams=%s\ninteraction=%s\nactivation=%s\n' "${convolution}" "${streams}" "${interaction}" "${activation}"
        printf 'slurm_job_id=%s\nhostname=%s\ngit_commit=%s\n' "${SLURM_JOB_ID}" "$(hostname)" "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
        printf 'source_manifest=%s\nworker_manifest=%s\nstarted_utc=%s\n' "${MANIFEST}" "${worker_manifest}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${attempt_dir}/run_info.txt"
    record_loaded_modules > "${attempt_dir}/modules.txt"

    local -a train_args=(
        --log-dir "${attempt_dir}/logs" train --manifest "${worker_manifest}" --runs-dir "${attempt_dir}"
        --device cpu --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}"
        --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" --num-workers "${NUM_WORKERS}"
        --patience "${PATIENCE}" --lr "${LEARNING_RATE}" --weight-decay "${WEIGHT_DECAY}" --seed "${seed}"
    )
    if [[ "${mode}" == "real" ]]; then
        "${VENV_DIR}/bin/python" -m prost_t2_classification "${train_args[@]}" --mode real
    else
        "${VENV_DIR}/bin/python" -m prost_t2_classification "${train_args[@]}" --mode complex \
            --complex-input-domain "${input_domain}" --complex-pooling "${pooling}" \
            --complex-normalization "${normalization}" --complex-convolution "${convolution}" \
            --complex-streams "${streams}" --complex-interaction "${interaction}" \
            --complex-activation "${activation}"
    fi
    "${VENV_DIR}/bin/python" - "${attempt_dir}" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
required = ("config.json", "history.csv", "threshold.json", "test_metrics.json")
runs = [p.parent for p in root.glob("*/config.json") if all((p.parent / name).is_file() for name in required)]
if len(runs) != 1:
    raise SystemExit(f"Expected one completed training run in {root}; found {len(runs)}")
PY
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${attempt_dir}/run_info.txt"
}

summarize_phase2() {
    rm -f -- "${EXPERIMENT_ROOT}/PHASE2_INCOMPLETE"
    if ! "${VENV_DIR}/bin/python" -m prost_t2_classification.experiment_summary \
        --experiment-root "${EXPERIMENT_ROOT}" --count "${PHASE2_SEEDS}" --seed-base "${PHASE2_SEED_BASE}"; then
        touch "${EXPERIMENT_ROOT}/PHASE2_INCOMPLETE"
        return 1
    fi
    touch "${EXPERIMENT_ROOT}/PHASE2_COMPLETE"
}

show_status() {
    [[ -x "${VENV_DIR}/bin/python" ]] || die "Missing experiment virtual environment; run check first."
    "${VENV_DIR}/bin/python" - "${EXPERIMENT_ROOT}" "${PILOT_SEED}" "${PHASE2_SEEDS}" "${PHASE2_SEED_BASE}" <<'PY'
from pathlib import Path
import sys
from prost_t2_classification.experiment_grid import experiment_model_specs

root = Path(sys.argv[1])
pilot_seed = int(sys.argv[2])
phase2_seeds = int(sys.argv[3])
seed_base = int(sys.argv[4])
models = experiment_model_specs()

def phase_status(phase: str, seeds: list[int]) -> None:
    completed = failed = 0
    for seed in seeds:
        for spec in models:
            model_dir = root / phase / f"seed_{seed}" / "models" / spec.model_key
            if (model_dir / "COMPLETE").is_file():
                completed += 1
            elif any(model_dir.glob("attempt_*/FAILED")):
                failed += 1
    expected = len(seeds) * len(models)
    pending = expected - completed - failed
    print(
        f"{phase}: expected={expected} completed={completed} "
        f"failed={failed} pending_or_queued={pending}"
    )

phase_status("phase1", [pilot_seed])
phase_status("phase2", [seed_base + index + 1 for index in range(phase2_seeds)])
print(f"phase2_complete_marker={(root / 'PHASE2_COMPLETE').is_file()}")
print(f"phase2_incomplete_marker={(root / 'PHASE2_INCOMPLETE').is_file()}")
PY
}

case "${1:-submit}" in
    submit) submit_experiment ;;
    check) check_host ;;
    status) show_status ;;
    resume-phase2) resume_phase2 ;;
    __worker) shift; check_configuration; run_worker "$@" ;;
    __summarize) summarize_phase2 ;;
    *) die "Usage: bash scripts/run_cluster_experiment.sh [check|submit|status|resume-phase2]" ;;
esac
