# Real vs Complex T2 Experiment — Laptop Edition

This branch runs the prepared-NPZ real-vs-complex experiment locally, without
Slurm or cluster scratch storage. It compares:

- a parameter-matched real CNN using four coil magnitudes;
- a phase-equivariant complex CNN using the same four complex coils.

The model, paired-seed design, validation-threshold tuning, and result tables are
the same as on `exp`. The local runner executes seeds sequentially, auto-selects
CUDA when available, uses laptop-safe memory and worker defaults, records the
local software/hardware environment, and can resume after interruption.

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
validation, and test split must contain both labels.

## Set up the laptop environment

Python 3.10 or newer is supported. In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --editable ".[dev]"
```

The normal PyTorch package gives a CPU build on many systems. If the laptop has
a supported NVIDIA GPU, install the appropriate CUDA-enabled PyTorch build for
the machine before the editable install. The runner reports the selected device
during its check.

## Check and run

Validate Python, dependencies, the device, branch, manifest, and all referenced
sample files:

```powershell
python scripts/run_local_experiment.py check
```

Run one pilot pair followed by three new paired seeds:

```powershell
python scripts/run_local_experiment.py run
```

This laptop profile keeps the cluster experiment's 20 epochs but uses batch size
2, gradient accumulation 16 (effective batch size 32), zero DataLoader worker
processes, and at most four CPU compute threads. Seeds run sequentially to avoid
memory contention and thermal overload. CUDA is used automatically when
available; force CPU with `--device cpu`.

To reproduce the full 20-seed phase from `exp` locally:

```powershell
python scripts/run_local_experiment.py run `
  --experiment-name laptop_maxpool_full `
  --phase2-seeds 20
```

Useful tuning options include `--epochs`, `--batch-size`,
`--gradient-accumulation-steps`, `--threads`, `--num-workers`, and `--device`.
Run `python scripts/run_local_experiment.py run --help` for the complete list.

The complex model supports three intermediate pooling modes: `max` (default),
`median` (select the full complex activation with median-ranked amplitude), and
`average` (average real and imaginary components). Select one with
`--complex-pooling max|median|average`. Each non-default mode receives a distinct
run-directory suffix so results cannot be mixed accidentally.

The k-space classifier transforms each prepared complex coil image with a
centered orthonormal FFT, then performs classification directly on the complex
k-space tensor without an inverse FFT. It has the same trainable parameter
count as the complex-image variants and always uses complex average pooling:

```powershell
python -m prost_t2_classification train `
  --manifest data/manifest.csv `
  --runs-dir runs/kspace_average `
  --mode complex_kspace `
  --device cpu `
  --epochs 20 `
  --batch-size 2 `
  --gradient-accumulation-steps 16
```

## Resume and results

Re-run the exact same command after an interruption. Completed seeds are
skipped, failed attempts are retained for diagnosis, and the first unfinished
seed is retried in a new attempt directory. Reusing an experiment name with a
different configuration is rejected to prevent mixed results.

The default output root is `runs/prost_t2_experiments/laptop_maxpool_v1`.
Per-seed checkpoints and logs are stored under `phase1/` and `phase2/`. Final
files are:

- `metrics_by_seed.csv`
- `summary_by_model.csv`
- `paired_deltas.csv`
- `paired_summary.csv`

To regenerate those tables without retraining:

```powershell
python scripts/run_local_experiment.py summarize
```
