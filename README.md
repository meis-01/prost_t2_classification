# Full-factorial complex T2 experiment

This branch contains a launch-ready CPU/Slurm experiment for prepared
four-coil prostate T2 NPZ data. It retains the original real baseline and the
three original complex pooling models, then expands the complex architecture
into a fixed factorial grid.

## Experiment grid

The launcher runs one real baseline plus 480 complex configurations:

| Factor | Levels |
|---|---|
| Input domain | image, k-space |
| Complex pooling | magnitude max, magnitude median, complex average |
| Complex normalization | RMSNorm, complex BatchNorm |
| Complex convolution | standard, widely-linear |
| Activation | modReLU, magnitude-gated SiLU, CReLU, cardioid |
| Streams and interaction | complex-only/none, complex-only/holographic, dual/none, dual/modulus-gate, dual/holographic |

The Cartesian product is `2 x 3 x 2 x 2 x 4 x 5 = 480` valid complex
configurations. Complex-only/modulus-gate is intentionally absent because a
modulus gate requires a real magnitude stream.

Channel widths are selected as a dependent variable for each architecture so
its trainable scalar parameter count stays within 1% of the fixed real
baseline. The selected widths and exact parameter count are written to every
run's `config.json`.

The original models keep these stable experiment keys and definitions:

- `real`: real magnitude baseline;
- `complex_max`: image/RMSNorm/standard/complex-only/none/modReLU with max pooling;
- `complex_median`: the same model with median pooling;
- `complex_average`: the same model with average pooling.

All other keys encode their factors, for example
`cx_ksp_avg_cbn_wl_dual_holo_card`.

## Prespecified design

- Technical pilot: seed `10383`, all 481 models.
- Confirmatory phase: seeds `24001` through `24004`, all 481 models per seed.
- Total seed count: five, including pilot seed `10383`.
- Pilot results are excluded from confirmatory summaries.
- Maximum epochs: 100; early-stopping patience: 8.
- Batch size: 8; gradient accumulation: 4; effective batch size: 32.
- AdamW learning rate: `1e-3`; weight decay: `1e-4`.
- Validation balanced accuracy selects each model's classification threshold.

The retained primary endpoint is the paired test-AUC difference
`complex_median - real`. All other grid comparisons are exploratory. The
summary reports paired differences, 10,000-resample percentile intervals, and
wins/ties for every complex configuration. The exact paired sign-flip test is
only run for the prespecified primary endpoint.

## Data

```text
data/
|-- manifest.csv
`-- samples/
    `-- *.npz
```

Each NPZ must contain a complex `image_complex` array shaped `(4, H, W)`. The
manifest requires `path`, `fastmri_pt_id`, `label`, `data_split`, and
`channels`. Training, validation, and test must each contain both classes, and
patient IDs must be disjoint across splits.

Image-domain complex inputs are phase-aligned and robustly magnitude-scaled.
K-space inputs additionally receive a centered orthonormal 2-D FFT before
scaling. Complex training inputs receive random global-phase augmentation.

## Cluster launch

On a CECI login node, update the branch and load Python 3.10-3.12:

```bash
git switch exp
git pull --ff-only origin exp
ml spider Python
ml load <selected-Python-module>
```

Run the preflight check. On its first run it creates the virtual environment
and installs the pinned dependencies:

```bash
bash scripts/run_cluster_experiment.sh check
```

Submit the pilot, sequential per-seed arrays, and dependent summary:

```bash
bash scripts/run_cluster_experiment.sh submit
```

The default launch creates:

- one 481-task pilot array with no launcher-side concurrency cap;
- 1,924 confirmatory tasks (`4 x 481`);
- 2,405 training tasks across all five seeds;
- exactly one 481-task array for each confirmatory seed;
- seed arrays chained in order, so all tasks for seed `24001` reach a terminal
  state before seed `24002` starts, continuing through seed `24004`;
- one summary job after the final seed array.

Every seed boundary, including the boundary after the pilot, uses `afterany`.
An individual model exception is recorded as `FAILED`, but it does not block
the remaining models in that array or any later seed. The summary also uses
`afterany`; if runs are missing, it writes `PHASE2_INCOMPLETE` instead of
`PHASE2_COMPLETE`. The preflight reads Slurm's `MaxArraySize` when available
and verifies that a 481-task array is supported.

By default, all 481 tasks in the active seed array are eligible to start at
once. Slurm, partition, and account/QoS limits determine the actual number of
simultaneous tasks. This uses the maximum cluster capacity available without
violating seed-by-seed ordering. Set `MAX_PARALLEL` to a positive number only
when a manual cap is desired; `MAX_PARALLEL=0` means unthrottled.

Defaults can be overridden, for example:

```bash
CPUS=20 MEMORY=48G TIME_LIMIT=48:00:00 MAX_PARALLEL=0 \
  bash scripts/run_cluster_experiment.sh submit
```

The checked-in defaults request the `work` partition, 16 CPUs, 32 GB per task,
and 48 hours per task. Set `TIME_LIMIT` explicitly if the target partition has
a different limit.

## Outputs and recovery

Persistent outputs default to:

```text
$GLOBALSCRATCH/prost_t2_experiments/complex_full_factorial_v1
```

Submission provenance includes:

- `model_grid.csv` and `model_grid.json`;
- `configuration.txt`, `environment.txt`, and `modules.txt`;
- `submission.txt` with every Slurm array and summary job ID.

Check completed, failed, and pending task counts at any time:

```bash
bash scripts/run_cluster_experiment.sh status
```

Each run lives under:

```text
phase1|phase2/seed_<seed>/models/<model-key>/attempt_<job>_<task>/
```

Successful attempts receive `SUCCESS` and their model directories receive
`COMPLETE`. Failed attempts are retained with `FAILED`, so diagnostics are not
overwritten.

Final aggregation writes `metrics_by_seed.csv`, `summary_by_model.csv`,
`paired_deltas.csv`, `paired_summary.csv`, `summary_metadata.json`, and
`PHASE2_COMPLETE`.

If phase two is interrupted, cancel any remaining original phase-two and
summary jobs, then run:

```bash
EXPERIMENT_ROOT=/absolute/path/to/the/experiment \
  bash scripts/run_cluster_experiment.sh resume-phase2
```

Recovery validates the original configuration, then resubmits the
confirmatory grid. Workers skip existing `COMPLETE` model/seed directories and
only rerun unfinished work. Pilot failures do not prevent phase-two recovery.
