# Real vs Complex-Pooling T2 Experiment

This branch contains the launch-ready, prepared-NPZ cluster experiment for a
paired four-model comparison:

- a parameter-matched real CNN using four coil magnitudes;
- a phase-equivariant complex CNN with magnitude-max pooling;
- the same complex CNN with median-amplitude selection pooling; and
- the same complex CNN with complex average pooling.

All complex variants use the same architecture outside the pooling operation.
Median pooling selects the activation with median-ranked amplitude while
retaining its full complex value and phase. Max pooling similarly retains the
maximum-amplitude complex activation. Average pooling averages the real and
imaginary components.

The prespecified primary question remains whether complex/median improves test
AUC relative to the real model. Complex/max and complex/average are retained as
exploratory pooling ablations and are run under the same paired design.

## Prespecified design

For each seed, all four models use the same dataset split, sampler seed,
optimizer settings, maximum epoch count, early-stopping rule, and Slurm
resource request. Each model resets all random-number generators to the seed
before training.

- Technical pilot: seed `10383`.
- Confirmatory seeds: `24001` through `24020`.
- The pilot gates the Slurm array but is excluded from confirmatory summaries.
- Maximum epochs: 100 for every model; early-stopping patience: 8 epochs.
- Batch size: 8; gradient accumulation: 4; effective batch size: 32.
- AdamW learning rate: `1e-3`; weight decay: `1e-4`.

The primary endpoint is the paired difference in test AUC:
`complex_median - real`. Secondary endpoints are test average precision,
balanced accuracy, sensitivity, and specificity. The max-vs-real and
average-vs-real results are explicitly exploratory. The classification
threshold is selected on validation balanced accuracy independently for every
trained model. Best validation AUC and epoch are diagnostics, not test
endpoints.

For every complex variant, the phase-two summary reports paired differences
against real, a 10,000-resample percentile bootstrap confidence interval, and
wins/ties. The two-sided paired sign-flip test is reserved for the prespecified
primary endpoint, complex/median minus real test AUC. With the default 20 seeds,
that test is enumerated exactly. These intervals describe training-seed
variability on the fixed test set; they are not confidence intervals for
sampling new patients.

## Data

Place the prepared dataset inside the clone:

```text
data/
|-- manifest.csv
`-- samples/
    `-- *.npz
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

The default workflow requests the `work` partition, 16 CPUs, 32 GB RAM, and 48
hours per model job, matching Lemaitre4's maximum job duration. Each seed/model
combination is a separate array task so the four variants do not share one
wall-time budget. Phase one is a four-task technical-pilot array. Only if every
pilot model succeeds, phase two submits 20 seeds x 4 models (80 array tasks),
with at most eight tasks running concurrently. A dependent summary job runs only
after every phase-two task succeeds. NPZ inputs are staged to `$LOCALSCRATCH` by
default.

Useful resource overrides can be supplied as environment variables:

```bash
CPUS=20 MEMORY=48G TIME_LIMIT=96:00:00 MAX_PARALLEL=4 \
  bash scripts/run_cluster_experiment.sh submit
```

The launcher is intentionally fixed to one real run and the three complex
pooling runs per seed. The model set and pooling methods are recorded in both
submission and per-job metadata. `EPOCHS=100` is the default; early stopping can
end a model before epoch 100 when validation performance stops improving.

## Outputs and monitoring

Persistent outputs default to:

```text
$GLOBALSCRATCH/prost_t2_experiments/pooling_vs_real_v1
```

The submission command prints the pilot, array, and summary job IDs. Monitor
them with the printed `squeue` command and inspect `slurm/*.out` and
`slurm/*.err`. A successful final aggregation writes:

- `metrics_by_seed.csv`: one row per seed and model;
- `summary_by_model.csv`: mean and standard deviation by model;
- `paired_deltas.csv`: seed-level differences for every complex pooling method
  relative to real;
- `paired_summary.csv`: confidence intervals and wins for all comparisons,
  plus the primary median-vs-real sign-flip result;
- `summary_metadata.json`: models, primary and exploratory comparisons,
  endpoints, seeds, resampling count, and inference scope;
- `configuration.txt`, `environment.txt`, and `modules.txt`: launch provenance;
- `PHASE2_COMPLETE`: terminal success marker.

The launcher refuses to reuse an existing experiment directory. Set a new,
descriptive `EXPERIMENT_NAME` for any intentional rerun.

### Recover phase two after completed pilots

If all four pilots produced complete run artifacts but an obsolete launcher
caused Slurm to mark the pilot array failed, cancel the blocked phase-two and
summary jobs, update the `exp` branch, and use the guarded recovery command:

```bash
scancel OLD_PHASE2_JOB OLD_SUMMARY_JOB
git pull --ff-only origin exp
EXPERIMENT_ROOT=/absolute/path/to/the/existing/experiment \
  bash scripts/run_cluster_experiment.sh resume-phase2
```

`resume-phase2` validates the original configuration and all four pilot
`config.json`, `history.csv`, `threshold.json`, and `test_metrics.json` files.
It refuses to run while the obsolete jobs are queued, when phase-two test
results already exist, or after a prior recovery. It then submits only the
20-seed, four-model phase-two array and its dependent summary job; the pilots
are not rerun. The replacement job IDs are recorded in `phase2_resume.txt`.
