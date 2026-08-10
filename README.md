# Real vs Complex T2 Experiment

This branch contains only the prepared-NPZ training experiment. It compares:

- a parameter-matched real CNN using four coil magnitudes;
- a phase-equivariant complex CNN using the same four complex coils.

The complex model uses complex convolution, complex RMS normalization, ModReLU,
and magnitude-max pooling that retains the selected location's full complex
value.

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
validation, and test split must contain both labels.

## CECI/Lemaitre4

Load an available Python 3.10, 3.11, or 3.12 module:

```bash
ml spider Python
ml load <selected-Python-module>
```

Validate the host, dependencies, data, and Slurm access:

```bash
bash scripts/run_cluster_experiment.sh check
```

Submit the experiment:

```bash
bash scripts/run_cluster_experiment.sh submit
```

Phase one runs one paired real/complex seed. If it succeeds, phase two runs 20
new paired seeds as a Slurm array, followed by an aggregation job. NPZ inputs are
staged to `$LOCALSCRATCH`; persistent outputs use
`$GLOBALSCRATCH/prost_t2_experiments/maxpool_v1` when `$GLOBALSCRATCH` exists.

Useful overrides can be supplied as environment variables:

```bash
CPUS=20 MEMORY=48G MAX_PARALLEL=4 bash scripts/run_cluster_experiment.sh submit
```

Final files include `metrics_by_seed.csv`, `summary_by_model.csv`,
`paired_deltas.csv`, and `paired_summary.csv`.
