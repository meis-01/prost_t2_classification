# Real vs Complex-Median T2 Experiment

This branch contains the launch-ready, prepared-NPZ training experiment for a
paired comparison between:

- a parameter-matched real CNN using four coil magnitudes; and
- a phase-equivariant complex CNN using the same four complex coils, with
  median-amplitude pooling that retains the selected activation's full complex
  value and phase.

The confirmatory question is whether the complex-median model improves test AUC
relative to the real model under the same training budget.

## Prespecified design

Each seed is a pair: the real and complex-median models use the same dataset
split, sampler seed, optimizer settings, epoch limit, early-stopping rule, and
Slurm allocation. Each model resets all random-number generators to the pair's
seed before training.

- Technical pilot: seed `10383`.
- Confirmatory seeds: `24001` through `24020`.
- The pilot gates the Slurm array but is excluded from confirmatory summaries.
- Maximum epochs: 20; early stopping patience: 8 epochs.
- Batch size: 8; gradient accumulation: 4; effective batch size: 32.
- AdamW learning rate: `1e-3`; weight decay: `1e-4`.

The primary endpoint is the paired difference in test AUC:
`complex_median - real`. Secondary endpoints are test average precision,
balanced accuracy, sensitivity, and specificity. The classification threshold
is selected on validation balanced accuracy independently for each trained
model. Best validation AUC and epoch are diagnostics, not test endpoints.

The phase-two summary reports paired differences, a 10,000-resample percentile
bootstrap confidence interval, wins/ties, and a two-sided paired sign-flip test
for the primary endpoint. With the default 20 seeds, the sign-flip test is
enumerated exactly. These intervals describe training-seed variability on the
fixed test set; they are not confidence intervals for sampling new patients.

## Data

Place the prepared dataset inside the clone:

```text
data/
├── manifest.csv
└── samples/
    └── *.npz
```

Each NPZ must contain `image_complex` with shape `(4, height, width)`. Manifest
paths are relative to `manifest.csv`. The manifest must contain `path`,
`fastmri_pt_id`, `label`, `data_split`, and `channels`; every training,
validation, and test split must contain both labels. Patient IDs must be
disjoint across splits.

## CECI/Lemaitre4 launch

Clone or update the repository and check out the `exp` branch. Load an available
Python 3.10, 3.11, or 3.12 module:

```bash
git switch exp
git pull --ff-only origin exp
ml spider Python
ml load <selected-Python-module>
```

Validate the branch, clean worktree, Python environment, CPU-only PyTorch,
prepared data, and Slurm access:

```bash
bash scripts/run_cluster_experiment.sh check
```

Submit the complete experiment once:

```bash
bash scripts/run_cluster_experiment.sh submit
```

The default workflow requests the `work` partition, 16 CPUs, 32 GB RAM, and 24
hours per paired job. Phase one runs the technical pilot. Only if it succeeds,
phase two submits the 20 paired seeds as a Slurm array with at most eight tasks
running concurrently. A dependent summary job runs only after every phase-two
pair succeeds. NPZ inputs are staged to `$LOCALSCRATCH` by default.

Useful resource overrides can be supplied as environment variables:

```bash
CPUS=20 MEMORY=48G TIME_LIMIT=36:00:00 MAX_PARALLEL=4 \
  bash scripts/run_cluster_experiment.sh submit
```

The model comparison is intentionally fixed: the launcher always runs only the
real model and the complex model with median pooling. Pooling is recorded in
both submission and per-job metadata.

## Outputs and monitoring

Persistent outputs default to:

```text
$GLOBALSCRATCH/prost_t2_experiments/median_vs_real_v1
```

The submission command prints the pilot, array, and summary job IDs. Monitor
them with the printed `squeue` command and inspect `slurm/*.out` and
`slurm/*.err`. A successful final aggregation writes:

- `metrics_by_seed.csv`: one row per seed and model;
- `summary_by_model.csv`: mean and standard deviation by model;
- `paired_deltas.csv`: seed-level `complex_median - real` differences;
- `paired_summary.csv`: confidence intervals, wins, and the primary sign-flip
  result;
- `summary_metadata.json`: endpoints, seeds, resampling count, and inference
  scope;
- `configuration.txt`, `environment.txt`, and `modules.txt`: launch provenance;
- `PHASE2_COMPLETE`: terminal success marker.

The launcher refuses to reuse an existing experiment directory. Set a new,
descriptive `EXPERIMENT_NAME` for any intentional rerun.
