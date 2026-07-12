# fastMRI Prostate T2 Coil Classification

This project builds a T2-only classification experiment for the fastMRI prostate
dataset. It follows the dataset paper's slice-level PI-RADS classification setup
and official patient split, but replaces the released RSS images with the
middle coil image from the middle T2 acquisition.

The experiment compares:

- a real-valued CNN that receives only coil image amplitudes;
- a complex-valued CNN that receives the same selected coil as a complex image.

The real CNN uses doubled channel widths to match the two-component complex
comparison: its channel widths are `64, 128, 256, 384`, compared with complex
widths `32, 64, 128, 192`.

PI-RADS labels are binarized as in the paper: `PI-RADS > 2` is clinically
significant prostate cancer.

## Install

From PyPI:

```powershell
python -m pip install prost-t2-classification
```

Confirm the console command is available:

```powershell
prost-t2 --help
```

For development from a local checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

For CUDA training, install the PyTorch build that matches your GPU/driver before
or after the editable install.

## Light Laptop Pipeline

Use `--light` to run the pipeline on 20 T2 MRI exams selected from the official
split labels: 10 training, 5 validation, and 5 test exams. Reconstruction keeps
only the middle acquisition and middle labeled slice from each selected exam;
NPZ preparation keeps the middle coil from that slice.

```powershell
prost-t2 run `
  --light `
  --download-script .\prostate_download_script.txt `
  --download-dir D:\fastmri_prostate_light\archives `
  --extract-dir D:\fastmri_prostate_light\raw `
  --recon-dir D:\fastmri_prostate_light\recon_t2 `
  --npz-dir D:\fastmri_prostate_light\npz_t2_middle_coil `
  --runs-dir D:\fastmri_prostate_light\runs `
  --epochs 1 `
  --batch-size 4
```

Equivalent helper script:

```powershell
.\scripts\run_light_pipeline.ps1 `
  -DownloadScript .\prostate_download_script.txt `
  -DownloadDir D:\fastmri_prostate_light\archives `
  -ExtractDir D:\fastmri_prostate_light\raw `
  -ReconDir D:\fastmri_prostate_light\recon_t2 `
  -NpzDir D:\fastmri_prostate_light\npz_t2_middle_coil `
  -RunsDir D:\fastmri_prostate_light\runs
```

Light mode downloads the labels first, chooses the 20 exams, and then downloads
only matching T2 H5 entries when the provided download script has one T2 curl
command per exam. If the script only exposes archive-level T2 tarballs, the
command exits before downloading the full T2 archives.

When training is enabled, the complex model uses `modrelu`.

If the raw T2 files and labels are already downloaded and expanded, skip the
download stage:

```powershell
prost-t2 run `
  --light `
  --skip-download `
  --extract-dir D:\fastmri_prostate\T2 `
  --labels D:\fastmri_prostate\labels `
  --recon-dir D:\fastmri_prostate_light\recon_t2 `
  --npz-dir D:\fastmri_prostate_light\npz_t2_middle_coil `
  --runs-dir D:\fastmri_prostate_light\runs `
  --epochs 1 `
  --batch-size 4
```

## Full Pipeline

For a full run using already downloaded and extracted data:

```powershell
prost-t2 run `
  --skip-download `
  --extract-dir D:\fastmri_prostate\T2 `
  --labels D:\fastmri_prostate\labels `
  --recon-dir D:\fastmri_prostate\recon_t2 `
  --npz-dir D:\fastmri_prostate\npz_t2_middle_coil `
  --runs-dir D:\fastmri_prostate\runs
```

Equivalent helper script:

```powershell
.\scripts\run_full_pipeline.ps1
```

To stop after reconstruction and NPZ preparation, skip training:

```powershell
prost-t2 run `
  --skip-download `
  --extract-dir D:\fastmri_prostate\T2 `
  --labels D:\fastmri_prostate\labels `
  --recon-dir D:\fastmri_prostate\recon_t2 `
  --npz-dir D:\fastmri_prostate\npz_t2_middle_coil `
  --skip-train
```

## Individual Stages

Download labels and T2 archives/files only:

```powershell
prost-t2 download --download-script .\prostate_download_script.txt --download-dir D:\fastmri_prostate\archives --no-extract
```

Download and extract labels plus T2 tarballs:

```powershell
prost-t2 download --download-script .\prostate_download_script.txt --download-dir D:\fastmri_prostate\archives --extract-dir D:\fastmri_prostate\raw
```

Run selected middle-acquisition GRAPPA/IFFT reconstruction with
`fastmri-tools`. Regular mode reconstructs all labeled slices; `--light`
reconstructs one middle labeled slice per exam:

```powershell
prost-t2 reconstruct --raw-root D:\fastmri_prostate\T2 --labels D:\fastmri_prostate\labels --recon-dir D:\fastmri_prostate\recon_t2
```

Create compact NPZ samples from the middle acquisition and middle coil:

```powershell
prost-t2 make-npz --labels D:\fastmri_prostate\labels --recon-dir D:\fastmri_prostate\recon_t2 --npz-dir D:\fastmri_prostate\npz_t2_middle_coil
```

Prepare NPZ files from extracted raw data without training:

```powershell
prost-t2 prepare-npz `
  --raw-root D:\fastmri_prostate\T2 `
  --labels D:\fastmri_prostate\labels `
  --recon-dir D:\fastmri_prostate\recon_t2 `
  --npz-dir D:\fastmri_prostate\npz_t2_middle_coil
```

Train both models:

```powershell
prost-t2 train --manifest D:\fastmri_prostate\npz_t2_middle_coil\manifest.csv --runs-dir D:\fastmri_prostate\runs --mode both
```

Train the complex model only:

```powershell
prost-t2 train --manifest D:\fastmri_prostate\npz_t2_middle_coil\manifest.csv --runs-dir D:\fastmri_prostate\runs --mode complex --complex-activation modrelu
```

## Data Decisions

- The official `data_split` column is used directly, and patient leakage across
  train/validation/test is checked before training.
- Before reconstruction, each selected T2 raw file is reduced to the middle
  acquisition (`shape[0] // 2`). Regular runs keep all labeled slices for the
  exam; `--light` keeps only the middle labeled slice.
- The selected k-space slices are reconstructed through `fastmri-tools` GRAPPA
  and centered IFFT primitives, producing a compact complex `image_complex`
  array with one acquisition, the selected slices, and the middle coil
  (`shape[2] // 2`).
- Each NPZ stores `image_complex` with shape `(coils, height, width)` plus
  patient, slice, split, acquisition, and coil metadata.
- After the best checkpoint is selected by validation AUC, the decision
  threshold is tuned on validation balanced accuracy and written to
  `threshold.json`; `test_metrics.json` uses that tuned threshold.

## Publishing

Releases are published to PyPI by GitHub Actions using PyPI Trusted Publishing,
so the repository does not need a long-lived PyPI API token.

Create a pending publisher on PyPI with these values:

- PyPI project name: `prost-t2-classification`
- GitHub owner: `meis-01`
- GitHub repository: `prost_t2_classification`
- Workflow file: `publish.yml`
- GitHub environment: `pypi`

Then bump `__version__` in `src/prost_t2_classification/__init__.py`, commit the
change, and push a matching tag:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

The publish workflow checks that the tag matches the package version before it
uploads the wheel and source distribution.
