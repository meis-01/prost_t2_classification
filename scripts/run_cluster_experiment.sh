#!/usr/bin/env bash
set -Eeuo pipefail

# Robust CPU/Slurm launcher for the full-factorial prostate T2 experiment.
# Intended for the cluster's `general` partition.
#
# Usage:
#   bash scripts/run_cluster_experiment.sh check
#   bash scripts/run_cluster_experiment.sh submit
#   bash scripts/run_cluster_experiment.sh status
#
# Important: commit this launcher before submit; the repository is required to be clean.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
MANIFEST="${MANIFEST:-${DATA_DIR}/manifest.csv}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-complex_full_factorial_v2}"
PERSISTENT_RUNS_ROOT="${PERSISTENT_RUNS_ROOT:-${GLOBALSCRATCH:-${REPO_ROOT}/runs}/prost_t2_experiments}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PERSISTENT_RUNS_ROOT}/${EXPERIMENT_NAME}}"
CLUSTER_REQUIREMENTS="${CLUSTER_REQUIREMENTS:-${REPO_ROOT}/requirements-cluster.txt}"

# Cluster settings chosen from observed accounting and node layout.
PARTITION="${PARTITION:-general}"
CPUS="${CPUS:-12}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-7-00:00:00}"
SLURM_HINT="${SLURM_HINT:-nomultithread}"

# Keep enough parallelism to use the cluster, while not launching hundreds of
# simultaneous data copies. Phase 2 uses one array per seed; the total cap is
# divided across those arrays.
PILOT_MAX_PARALLEL="${PILOT_MAX_PARALLEL:-48}"
PHASE2_TOTAL_MAX_PARALLEL="${PHASE2_TOTAL_MAX_PARALLEL:-48}"

PILOT_SEED="${PILOT_SEED:-10383}"
PHASE2_SEEDS="${PHASE2_SEEDS:-4}"
PHASE2_SEED_BASE="${PHASE2_SEED_BASE:-24000}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PATIENCE="${PATIENCE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
USE_SYSTEM_TORCH="${USE_SYSTEM_TORCH:-auto}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${GLOBALSCRATCH:-${HOME}}/.cache/pip-prost-t2}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-exp}"

# Keep local staging because every epoch repeatedly opens individual NPZ files.
# A small randomized delay avoids 48 tasks hitting shared storage at the same instant.
STAGE_DATA="${STAGE_DATA:-1}"
STAGE_JITTER_MAX="${STAGE_JITTER_MAX:-30}"

MODEL_COUNT=481
COMPLEX_MODEL_COUNT=480
PRIMARY_ENDPOINT="test_auc"
PRIMARY_COMPARISON="complex_median_minus_real"

SUBMITTED_PHASE2_ARRAY_JOBS=""
SUBMITTED_SUMMARY_JOB=""


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
    require_positive_integer PILOT_MAX_PARALLEL "${PILOT_MAX_PARALLEL}"
    require_positive_integer PHASE2_TOTAL_MAX_PARALLEL "${PHASE2_TOTAL_MAX_PARALLEL}"
    require_positive_integer PHASE2_SEEDS "${PHASE2_SEEDS}"
    require_positive_integer EPOCHS "${EPOCHS}"
    require_positive_integer BATCH_SIZE "${BATCH_SIZE}"
    require_positive_integer GRADIENT_ACCUMULATION_STEPS "${GRADIENT_ACCUMULATION_STEPS}"
    require_positive_integer PATIENCE "${PATIENCE}"
    [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || die "NUM_WORKERS must be a non-negative integer."
    (( NUM_WORKERS < CPUS )) || die "NUM_WORKERS must be smaller than CPUS."
    [[ "${STAGE_DATA}" =~ ^[01]$ ]] || die "STAGE_DATA must be 0 or 1."
    [[ "${STAGE_JITTER_MAX}" =~ ^[0-9]+$ ]] || die "STAGE_JITTER_MAX must be a non-negative integer."
    [[ "${USE_SYSTEM_TORCH}" =~ ^(auto|0|1)$ ]] || die "USE_SYSTEM_TORCH must be auto, 0, or 1."
}

array_spec() {
    local count="${1:?count required}" cap="${2:?parallel cap required}"
    printf '0-%s%%%s\n' "$((count - 1))" "${cap}"
}

phase2_per_seed_cap() {
    local cap=$((PHASE2_TOTAL_MAX_PARALLEL / PHASE2_SEEDS))
    (( cap >= 1 )) || cap=1
    printf '%s\n' "${cap}"
}

check_repository() {
    command -v git >/dev/null 2>&1 || die "git is not available."
    local branch
    branch="$(git -C "${REPO_ROOT}" branch --show-current)"
    [[ "${branch}" == "${EXPECTED_BRANCH}" ]] || \
        die "Expected branch ${EXPECTED_BRANCH}; current branch is ${branch:-detached}."
    git -C "${REPO_ROOT}" diff --quiet || die "Tracked files have unstaged changes. Commit the launcher/code first."
    git -C "${REPO_ROOT}" diff --cached --quiet || die "Tracked files have staged changes. Commit them first."
}

check_slurm() {
    command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable on this host."
    command -v sinfo >/dev/null 2>&1 || die "sinfo is unavailable on this host."
    sinfo -h -p "${PARTITION}" >/dev/null 2>&1 || die "Slurm partition ${PARTITION} is unavailable."

    if command -v scontrol >/dev/null 2>&1; then
        local max_array_size
        max_array_size="$(scontrol show config 2>/dev/null | awk -F= '
            /MaxArraySize/ && !found {
                gsub(/[[:space:]]/, "", $2); print $2; found=1
            }')"
        if [[ "${max_array_size}" =~ ^[1-9][0-9]*$ ]]; then
            (( MODEL_COUNT <= max_array_size )) || \
                die "Need ${MODEL_COUNT} array entries; cluster MaxArraySize is ${max_array_size}."
        fi
    fi
}

ensure_environment() {
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "${PYTHON_BIN} is unavailable."
    [[ -f "${CLUSTER_REQUIREMENTS}" ]] || die "Missing ${CLUSTER_REQUIREMENTS}."

    "${PYTHON_BIN}" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
    raise SystemExit(f"Python {sys.version.split()[0]} is unsupported; use Python 3.10-3.12")
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
        die "USE_SYSTEM_TORCH=1 but loaded Python has no compatible CPU-only PyTorch."
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

    "${VENV_DIR}/bin/python" -m pip install -q 'pip==24.3.1' 'setuptools==75.6.0' 'wheel==0.45.1'
    "${VENV_DIR}/bin/python" -m pip install -q --requirement "${CLUSTER_REQUIREMENTS}"
    if [[ "${environment_mode}" != "system-torch" ]]; then
        "${VENV_DIR}/bin/python" -m pip install -q "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
    fi
    "${VENV_DIR}/bin/python" -m pip install -q --no-deps --editable "${REPO_ROOT}"

    "${VENV_DIR}/bin/python" - <<'PY'
import torch
from prost_t2_classification.experiment_grid import experiment_model_specs
if torch.version.cuda is not None:
    raise SystemExit(f"Expected CPU-only PyTorch, found CUDA {torch.version.cuda}")
if len(experiment_model_specs()) != 481:
    raise SystemExit("Experiment grid does not contain exactly 481 models")
print(f"PyTorch {torch.__version__}; grid=481 models")
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
    raise SystemExit(f"{first} must contain complex image_complex shaped (4, H, W)")
print(f"Validated {len(frame)} samples and all manifest paths")
PY
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
    export LEARNING_RATE WEIGHT_DECAY STAGE_DATA STAGE_JITTER_MAX
    export PRIMARY_ENDPOINT PRIMARY_COMPARISON
}

write_submission_metadata() {
    mkdir -p "${EXPERIMENT_ROOT}"
    "${VENV_DIR}/bin/python" -m prost_t2_classification.experiment_grid write \
        --csv "${EXPERIMENT_ROOT}/model_grid.csv" \
        --json "${EXPERIMENT_ROOT}/model_grid.json"

    local per_seed_cap
    per_seed_cap="$(phase2_per_seed_cap)"
    {
        printf 'git_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
        printf 'manifest=%s\n' "${MANIFEST}"
        printf 'experiment_root=%s\n' "${EXPERIMENT_ROOT}"
        printf 'model_count=%s\n' "${MODEL_COUNT}"
        printf 'complex_model_count=%s\n' "${COMPLEX_MODEL_COUNT}"
        printf 'pilot_seed=%s\n' "${PILOT_SEED}"
        printf 'phase2_seeds=%s\n' "${PHASE2_SEEDS}"
        printf 'phase2_seed_base=%s\n' "${PHASE2_SEED_BASE}"
        printf 'phase2_task_count=%s\n' "$((PHASE2_SEEDS * MODEL_COUNT))"
        printf 'pilot_to_phase2_dependency=afterok\n'
        printf 'phase2_execution=parallel_seed_arrays\n'
        printf 'summary_dependency=afterany_all_phase2_arrays\n'
        printf 'partition=%s\ncpus=%s\nmemory=%s\ntime_limit=%s\n' \
            "${PARTITION}" "${CPUS}" "${MEMORY}" "${TIME_LIMIT}"
        printf 'pilot_max_parallel=%s\n' "${PILOT_MAX_PARALLEL}"
        printf 'phase2_total_max_parallel=%s\n' "${PHASE2_TOTAL_MAX_PARALLEL}"
        printf 'phase2_per_seed_max_parallel=%s\n' "${per_seed_cap}"
        printf 'epochs=%s\nbatch_size=%s\ngradient_accumulation_steps=%s\n' \
            "${EPOCHS}" "${BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}"
        printf 'num_workers=%s\npatience=%s\nlearning_rate=%s\nweight_decay=%s\n' \
            "${NUM_WORKERS}" "${PATIENCE}" "${LEARNING_RATE}" "${WEIGHT_DECAY}"
        printf 'stage_data=%s\nstage_jitter_max=%s\n' "${STAGE_DATA}" "${STAGE_JITTER_MAX}"
        printf 'torch_version_request=%s\n' "${TORCH_VERSION}"
    } > "${EXPERIMENT_ROOT}/configuration.txt"

    "${VENV_DIR}/bin/python" -m pip freeze > "${EXPERIMENT_ROOT}/environment.txt"
    record_loaded_modules > "${EXPERIMENT_ROOT}/modules.txt"
}

check_host() {
    check_configuration
    check_repository
    check_slurm
    ensure_environment
    validate_data
    local per_seed_cap
    per_seed_cap="$(phase2_per_seed_cap)"
    printf '\nHost check passed.\n'
    printf 'Partition: %s | CPUs/model: %s | RAM/model: %s | wall-time: %s\n' \
        "${PARTITION}" "${CPUS}" "${MEMORY}" "${TIME_LIMIT}"
    printf 'Pilot: %s tasks, max %s concurrent.\n' "${MODEL_COUNT}" "${PILOT_MAX_PARALLEL}"
    printf 'Phase 2: %s seed arrays x %s tasks, max %s/seed (~%s total).\n' \
        "${PHASE2_SEEDS}" "${MODEL_COUNT}" "${per_seed_cap}" "$((per_seed_cap * PHASE2_SEEDS))"
}

submit_phase2_jobs() {
    local dependency_job="${1:?pilot dependency required}" submission_file="${2:?submission file required}"
    local seed_index seed array_job summary_job per_seed_cap dep_ids
    local -a phase2_ids=()
    per_seed_cap="$(phase2_per_seed_cap)"

    for ((seed_index = 0; seed_index < PHASE2_SEEDS; seed_index++)); do
        seed=$((PHASE2_SEED_BASE + seed_index + 1))
        array_job="$(sbatch --parsable \
            --job-name="prost-grid-s${seed}" --partition="${PARTITION}" \
            --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" --mem="${MEMORY}" \
            --time="${TIME_LIMIT}" --hint="${SLURM_HINT}" \
            --array="$(array_spec "${MODEL_COUNT}" "${per_seed_cap}")" \
            --dependency="afterok:${dependency_job}" --kill-on-invalid-dep=yes \
            --chdir="${REPO_ROOT}" \
            --output="${EXPERIMENT_ROOT}/slurm/phase2_seed${seed}_%A_%a.out" \
            --error="${EXPERIMENT_ROOT}/slurm/phase2_seed${seed}_%A_%a.err" \
            --export="ALL,PHASE2_SEED_INDEX=${seed_index}" \
            "${SCRIPT_PATH}" __worker phase2)"
        array_job="${array_job%%;*}"
        phase2_ids+=("${array_job}")
        printf 'phase2_seed_%s_job=%s\n' "${seed}" "${array_job}" | tee -a "${submission_file}"
    done

    SUBMITTED_PHASE2_ARRAY_JOBS="$(IFS=,; printf '%s' "${phase2_ids[*]}")"
    printf 'phase2_array_jobs=%s\n' "${SUBMITTED_PHASE2_ARRAY_JOBS}" | tee -a "${submission_file}"

    dep_ids="$(IFS=:; printf '%s' "${phase2_ids[*]}")"
    summary_job="$(sbatch --parsable \
        --job-name=prost-grid-summary --partition="${PARTITION}" --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --mem=8G --time=02:00:00 \
        --dependency="afterany:${dep_ids}" \
        --chdir="${REPO_ROOT}" \
        --output="${EXPERIMENT_ROOT}/slurm/summary_%j.out" \
        --error="${EXPERIMENT_ROOT}/slurm/summary_%j.err" --export=ALL \
        "${SCRIPT_PATH}" __summarize)"
    SUBMITTED_SUMMARY_JOB="${summary_job%%;*}"
    printf 'summary_job=%s\n' "${SUBMITTED_SUMMARY_JOB}" | tee -a "${submission_file}"
}

submit_experiment() {
    check_configuration
    check_repository
    check_slurm
    [[ ! -e "${EXPERIMENT_ROOT}/submission.txt" ]] || \
        die "Experiment already submitted; choose a new EXPERIMENT_NAME."
    [[ ! -e "${EXPERIMENT_ROOT}/submission.partial" ]] || \
        die "Partial submission exists; inspect it before retrying."

    ensure_environment
    validate_data
    mkdir -p "${EXPERIMENT_ROOT}/slurm" "${EXPERIMENT_ROOT}/phase1" "${EXPERIMENT_ROOT}/phase2"
    write_submission_metadata
    export_worker_environment

    local pilot_job submission_partial
    submission_partial="${EXPERIMENT_ROOT}/submission.partial"
    pilot_job="$(sbatch --parsable \
        --job-name=prost-grid-pilot --partition="${PARTITION}" \
        --nodes=1 --ntasks=1 --cpus-per-task="${CPUS}" --mem="${MEMORY}" \
        --time="${TIME_LIMIT}" --hint="${SLURM_HINT}" \
        --array="$(array_spec "${MODEL_COUNT}" "${PILOT_MAX_PARALLEL}")" \
        --chdir="${REPO_ROOT}" \
        --output="${EXPERIMENT_ROOT}/slurm/phase1_%A_%a.out" \
        --error="${EXPERIMENT_ROOT}/slurm/phase1_%A_%a.err" --export=ALL \
        "${SCRIPT_PATH}" __worker phase1)"
    pilot_job="${pilot_job%%;*}"
    printf 'phase1_job=%s\n' "${pilot_job}" | tee "${submission_partial}"

    # All confirmatory seed arrays wait for a completely successful pilot,
    # then become eligible at the same time. They are NOT chained seed-to-seed.
    submit_phase2_jobs "${pilot_job}" "${submission_partial}"

    mv -- "${submission_partial}" "${EXPERIMENT_ROOT}/submission.txt"
    cat "${EXPERIMENT_ROOT}/submission.txt"
    printf '\nMonitor with:\n  squeue -j %s,%s,%s\n' \
        "${pilot_job}" "${SUBMITTED_PHASE2_ARRAY_JOBS}" "${SUBMITTED_SUMMARY_JOB}"
}

# Globals intentionally used only inside a worker subshell.
STAGED_DATA_DIR=""
WORKER_MANIFEST=""

stage_worker_data() {
    if [[ "${STAGE_DATA}" == "0" ]]; then
        WORKER_MANIFEST="${MANIFEST}"
        return 0
    fi

    local scratch_root="${LOCALSCRATCH:-${TMPDIR:-}}"
    [[ -n "${scratch_root}" && -d "${scratch_root}" ]] || \
        die "STAGE_DATA=1 but no LOCALSCRATCH or TMPDIR exists."

    if (( STAGE_JITTER_MAX > 0 )); then
        sleep "$((RANDOM % (STAGE_JITTER_MAX + 1)))"
    fi

    STAGED_DATA_DIR="${scratch_root%/}/prost_t2_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    rm -rf -- "${STAGED_DATA_DIR}"
    mkdir -p "${STAGED_DATA_DIR}"
    cp -a "${DATA_DIR}/." "${STAGED_DATA_DIR}/"
    [[ -f "${STAGED_DATA_DIR}/manifest.csv" ]] || die "Data staging did not produce manifest.csv."
    WORKER_MANIFEST="${STAGED_DATA_DIR}/manifest.csv"
}

# IMPORTANT: this is a subshell function. The EXIT trap therefore runs before
# function-local marker paths go out of scope. This fixes the original
# `failure_marker: unbound variable` bug under `set -u`.
run_worker() (
    local phase="${1:?worker phase is required}"
    local local_task seed_index seed model_index
    local model_key mode input_domain pooling normalization convolution streams interaction activation

    [[ -n "${SLURM_JOB_ID:-}" ]] || die "Worker must run inside Slurm."
    [[ -x "${VENV_DIR}/bin/python" ]] || die "Missing experiment virtual environment."
    local_task="${SLURM_ARRAY_TASK_ID:?array task id is required}"

    if [[ "${phase}" == "phase1" ]]; then
        seed="${PILOT_SEED}"
        model_index="${local_task}"
    elif [[ "${phase}" == "phase2" ]]; then
        seed_index="${PHASE2_SEED_INDEX:?phase2 requires PHASE2_SEED_INDEX}"
        (( seed_index < PHASE2_SEEDS )) || die "Phase-two seed index out of range."
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
        return 0
    fi

    local attempt_dir="${model_dir}/attempt_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    local failure_marker="${attempt_dir}/FAILED"
    local success_marker="${attempt_dir}/SUCCESS"
    local completion_marker="${model_dir}/COMPLETE"
    mkdir -p "${attempt_dir}/logs"

    STAGED_DATA_DIR=""
    WORKER_MANIFEST=""

    worker_exit() {
        local exit_status=$? tmp_marker
        trap - EXIT INT TERM

        # Local scratch is disposable and should never accumulate across jobs.
        if [[ -n "${STAGED_DATA_DIR:-}" ]]; then
            rm -rf -- "${STAGED_DATA_DIR}" || true
        fi

        if (( exit_status == 0 )); then
            tmp_marker="${completion_marker}.tmp.${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}"
            if rm -f -- "${failure_marker}" \
                && touch "${success_marker}" \
                && printf 'job_id=%s\ncompleted_utc=%s\n' \
                    "${SLURM_JOB_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${tmp_marker}" \
                && mv -f -- "${tmp_marker}" "${completion_marker}"; then
                :
            else
                exit_status=1
                rm -f -- "${success_marker}" "${completion_marker}" "${tmp_marker:-}" || true
                printf 'exit_code=%s\nreason=Could not write success markers\n' \
                    "${exit_status}" > "${failure_marker}" || true
            fi
        else
            rm -f -- "${success_marker}" || true
            printf 'exit_code=%s\nfailed_utc=%s\n' \
                "${exit_status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${failure_marker}" || true
        fi
        exit "${exit_status}"
    }

    # Explicit signal exits give useful failure codes; EXIT performs bookkeeping.
    trap 'exit 130' INT
    trap 'exit 143' TERM
    trap worker_exit EXIT

    local allocated_cpus="${SLURM_CPUS_PER_TASK:-${CPUS}}"
    local torch_threads=$((allocated_cpus - NUM_WORKERS))
    (( torch_threads >= 1 )) || torch_threads=1

    export OMP_NUM_THREADS="${torch_threads}"
    export MKL_NUM_THREADS="${torch_threads}"
    export OPENBLAS_NUM_THREADS="${torch_threads}"
    export NUMEXPR_NUM_THREADS="${torch_threads}"
    export OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
    export MALLOC_ARENA_MAX=4 PYTHONUNBUFFERED=1

    stage_worker_data

    local cpu_model="unknown"
    if command -v lscpu >/dev/null 2>&1; then
        cpu_model="$(lscpu | awk -F: '/Model name/ {sub(/^[[:space:]]+/, "", $2); print $2; exit}')"
        lscpu > "${attempt_dir}/hardware.txt" || true
    fi

    {
        printf 'phase=%s\nseed=%s\nmodel_index=%s\nmodel_key=%s\n' \
            "${phase}" "${seed}" "${model_index}" "${model_key}"
        printf 'mode=%s\ninput_domain=%s\npooling=%s\nnormalization=%s\n' \
            "${mode}" "${input_domain}" "${pooling}" "${normalization}"
        printf 'convolution=%s\nstreams=%s\ninteraction=%s\nactivation=%s\n' \
            "${convolution}" "${streams}" "${interaction}" "${activation}"
        printf 'slurm_job_id=%s\nslurm_array_job_id=%s\nslurm_array_task_id=%s\n' \
            "${SLURM_JOB_ID}" "${SLURM_ARRAY_JOB_ID:-}" "${SLURM_ARRAY_TASK_ID}"
        printf 'hostname=%s\ncpu_model=%s\nallocated_cpus=%s\ntorch_threads=%s\nnum_workers=%s\n' \
            "$(hostname)" "${cpu_model}" "${allocated_cpus}" "${torch_threads}" "${NUM_WORKERS}"
        printf 'git_commit=%s\n' "$(git -C "${REPO_ROOT}" rev-parse HEAD)"
        printf 'source_manifest=%s\nworker_manifest=%s\nstarted_utc=%s\n' \
            "${MANIFEST}" "${WORKER_MANIFEST}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "${attempt_dir}/run_info.txt"
    record_loaded_modules > "${attempt_dir}/modules.txt"

    local -a train_args=(
        --log-dir "${attempt_dir}/logs"
        train
        --manifest "${WORKER_MANIFEST}"
        --runs-dir "${attempt_dir}"
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
        "${VENV_DIR}/bin/python" -m prost_t2_classification "${train_args[@]}" --mode real
    else
        "${VENV_DIR}/bin/python" -m prost_t2_classification "${train_args[@]}" --mode complex \
            --complex-input-domain "${input_domain}" \
            --complex-pooling "${pooling}" \
            --complex-normalization "${normalization}" \
            --complex-convolution "${convolution}" \
            --complex-streams "${streams}" \
            --complex-interaction "${interaction}" \
            --complex-activation "${activation}"
    fi

    # Do not mark COMPLETE unless exactly one finished training run has all
    # expected result files.
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
)

summarize_phase2() {
    rm -f -- "${EXPERIMENT_ROOT}/PHASE2_INCOMPLETE" "${EXPERIMENT_ROOT}/PHASE2_COMPLETE"
    if ! "${VENV_DIR}/bin/python" -m prost_t2_classification.experiment_summary \
        --experiment-root "${EXPERIMENT_ROOT}" \
        --count "${PHASE2_SEEDS}" \
        --seed-base "${PHASE2_SEED_BASE}"; then
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
    print(f"{phase}: expected={expected} completed={completed} failed={failed} pending_or_queued={pending}")

phase_status("phase1", [pilot_seed])
phase_status("phase2", [seed_base + i + 1 for i in range(phase2_seeds)])
print(f"phase2_complete_marker={(root / 'PHASE2_COMPLETE').is_file()}")
print(f"phase2_incomplete_marker={(root / 'PHASE2_INCOMPLETE').is_file()}")
PY
}

case "${1:-submit}" in
    check) check_host ;;
    submit) submit_experiment ;;
    status) show_status ;;
    __worker) shift; check_configuration; run_worker "$@" ;;
    __summarize) summarize_phase2 ;;
    *) die "Usage: bash scripts/run_cluster_experiment.sh [check|submit|status]" ;;
esac
