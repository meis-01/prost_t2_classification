# fastMRI Prostate T2 Coil Classification

This project builds a T2-only classification experiment for the fastMRI prostate
dataset. It follows the dataset paper's slice-level PI-RADS classification setup
and official patient split, but replaces the released RSS images with selected
coil images from the middle T2 acquisition.

The experiment compares:

- a real-valued CNN that receives only coil image amplitudes;
- a complex-valued CNN that receives the same selected coils as complex images.

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

## Full Pipeline

The full command prompts for storage locations if you omit them:

```powershell
prost-t2 run --download-script .\prostate_download_script.txt
```

Equivalent non-interactive form:

```powershell
prost-t2 run `
  --download-script .\prostate_download_script.txt `
  --download-dir D:\fastmri_prostate\archives `
  --extract-dir D:\fastmri_prostate\raw `
  --recon-dir D:\fastmri_prostate\recon_t2 `
  --npz-dir D:\fastmri_prostate\npz_t2_coils `
  --runs-dir D:\fastmri_prostate\runs
```

To stop after downloading, reconstruction, and NPZ preparation, skip training:

```powershell
prost-t2 run `
  --download-script .\prostate_download_script.txt `
  --download-dir D:\fastmri_prostate\archives `
  --extract-dir D:\fastmri_prostate\raw `
  --recon-dir D:\fastmri_prostate\recon_t2 `
  --npz-dir D:\fastmri_prostate\npz_t2_coils `
  --skip-train
```

## Individual Stages

Download labels and T2 tarballs only:

```powershell
prost-t2 download --download-script .\prostate_download_script.txt --download-dir D:\fastmri_prostate\archives --no-extract
```

Download and extract labels plus T2 tarballs:

```powershell
prost-t2 download --download-script .\prostate_download_script.txt --download-dir D:\fastmri_prostate\archives --extract-dir D:\fastmri_prostate\raw
```

Run GRAPPA/IFFT reconstruction with `fastmri-tools`:

```powershell
prost-t2 reconstruct --raw-root D:\fastmri_prostate\raw --recon-dir D:\fastmri_prostate\recon_t2
```

Create compact NPZ samples from the middle acquisition and top-energy coils:

```powershell
prost-t2 make-npz --labels D:\fastmri_prostate\raw --recon-dir D:\fastmri_prostate\recon_t2 --npz-dir D:\fastmri_prostate\npz_t2_coils
```

Prepare NPZ files from extracted raw data without training:

```powershell
prost-t2 prepare-npz `
  --raw-root D:\fastmri_prostate\raw `
  --recon-dir D:\fastmri_prostate\recon_t2 `
  --npz-dir D:\fastmri_prostate\npz_t2_coils
```

Train both models:

```powershell
prost-t2 train --manifest D:\fastmri_prostate\npz_t2_coils\manifest.csv --runs-dir D:\fastmri_prostate\runs --mode both
```

## Data Decisions

- The official `data_split` column is used directly, and patient leakage across
  train/validation/test is checked before training.
- T2 `kspace` is reconstructed through `fastmri-tools`, producing complex
  `image_complex` arrays.
- The acquisition dimension is reduced by selecting the middle acquisition
  (`shape[0] // 2`).
- Up to five coils are selected per patient volume using highest image-space
  energy, measured on the selected acquisition across all slices.
- NPZ files store `image_complex` with shape `(coils, height, width)` plus
  patient, slice, split, and coil metadata.

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
