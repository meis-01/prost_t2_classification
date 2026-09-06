"""Generate the data panels for manuscript Figure 1.

The figure uses one deterministic training-split sample and the preprocessing
implemented by ``prost_t2_classification.dataset``/``image_ops``.  The Python
side writes only raster panel images and metadata; panel assembly is kept in
the companion TikZ document.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUTPUT_DIR.parents[2]
DEFAULT_MANIFEST = Path(r"C:\Users\meisa\Data\npz_t2_phase4\manifest.csv")
PHASE_WEIGHT_LAMBDA = 4.0
PHASE_WEIGHT_GAMMA = 1.5
FOURIER_PHASE_CROP_FRACTION = 0.5

# Import the repository implementation rather than reproducing its numerical
# operations here.  This script is an output generator, not experiment code.
sys.path.insert(0, str(REPO_ROOT / "src"))
from prost_t2_classification.dataset import validate_manifest  # noqa: E402
from prost_t2_classification.image_ops import (  # noqa: E402
    align_multicoil_phase,
    centered_fft2,
    scale_complex_by_magnitude,
)


def _support_mask(image: np.ndarray, *, background_threshold: float = 0.02) -> np.ndarray:
    """Return the RSS support used by ``align_multicoil_phase``."""

    rss = np.sqrt(np.sum(np.abs(image) ** 2, axis=0))
    signal_scale = float(np.percentile(rss, 99.0))
    if signal_scale <= 1e-6:
        return np.zeros(rss.shape, dtype=bool)
    return rss >= background_threshold * signal_scale


def _magnitude_weight(
    complex_representation: np.ndarray,
    *,
    lambda_value: float,
    gamma: float,
) -> tuple[np.ndarray, float]:
    """Compute a continuous display-only magnitude weight."""

    log_magnitude = np.log1p(lambda_value * np.abs(complex_representation))
    percentile_99 = float(np.percentile(log_magnitude, 99.0))
    if percentile_99 <= 1e-6:
        return np.zeros(log_magnitude.shape, dtype=np.float32), percentile_99
    weight = np.clip(log_magnitude / percentile_99, 0.0, 1.0) ** gamma
    return weight.astype(np.float32), percentile_99


def _center_crop(array: np.ndarray, fraction: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = array.shape[-2:]
    crop_height = max(1, int(round(height * fraction)))
    crop_width = max(1, int(round(width * fraction)))
    y0 = (height - crop_height) // 2
    x0 = (width - crop_width) // 2
    y1 = y0 + crop_height
    x1 = x0 + crop_width
    return array[..., y0:y1, x0:x1], (y0, y1, x0, x1)


def _select_training_row(manifest: pd.DataFrame) -> pd.Series:
    """Select the lexicographically smallest training path, stably."""

    training = manifest[
        manifest["data_split"].astype(str).str.strip().str.lower().eq("training")
    ].copy()
    if training.empty:
        raise ValueError("The manifest has no training-split rows.")
    return training.sort_values(
        ["path", "fastmri_pt_id", "slice_index"], kind="mergesort"
    ).iloc[0]


def _load_image(sample_path: Path) -> np.ndarray:
    with np.load(sample_path) as npz:
        if "image_complex" not in npz:
            raise ValueError(f"{sample_path} does not contain image_complex.")
        raw_image = npz["image_complex"]
    if not np.iscomplexobj(raw_image):
        raise ValueError(f"{sample_path} image_complex must use a complex dtype.")
    image = raw_image.astype(np.complex64)
    if image.ndim != 3 or image.shape[0] != 4:
        raise ValueError(f"{sample_path} image_complex must have shape (4, height, width).")
    if min(image.shape[1:]) < 8 or not np.isfinite(image).all():
        raise ValueError(f"{sample_path} image_complex has invalid spatial dimensions or values.")
    return image


def _save_panel(
    array: np.ndarray,
    path: Path,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    mask: np.ndarray | None = None,
    phase_weight: np.ndarray | None = None,
) -> None:
    rendered = np.asarray(array, dtype=np.float32).copy()
    cmap_obj = plt.get_cmap(cmap).copy()
    fig = plt.figure(figsize=(3.5, 3.5), dpi=600, frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    if phase_weight is not None:
        weight = np.clip(np.asarray(phase_weight, dtype=np.float32), 0.0, 1.0)
        if mask is not None:
            weight = np.where(mask, weight, 0.0)
        phase_color = cmap_obj(np.clip((rendered - vmin) / (vmax - vmin), 0.0, 1.0))[..., :3]
        neutral = np.ones_like(phase_color)
        rgb = phase_color * weight[..., None] + neutral * (1.0 - weight[..., None])
        rgba = np.concatenate((rgb, np.ones((*rgb.shape[:2], 1), dtype=np.float32)), axis=-1)
        ax.imshow(rgba, origin="lower", interpolation="nearest")
    else:
        if mask is not None:
            rendered[~mask] = np.nan
            cmap_obj.set_bad((1.0, 1.0, 1.0, 0.0))
        ax.imshow(rendered, cmap=cmap_obj, vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(path, dpi=600, transparent=True, pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = pd.read_csv(manifest_path)
    validate_manifest(manifest)
    selected_row = _select_training_row(manifest)
    sample_path = (manifest_path.parent / str(selected_row["path"])).resolve()
    if not sample_path.is_file():
        raise FileNotFoundError(sample_path)

    raw_image = _load_image(sample_path)
    if raw_image.shape[0] != 4:
        raise ValueError("Figure 1 requires exactly four complex receiver coils.")

    aligned_image = align_multicoil_phase(raw_image)
    image_support = _support_mask(aligned_image)
    image_scaled = scale_complex_by_magnitude(aligned_image, shared_scale=True)

    fourier_complex = centered_fft2(aligned_image)
    fourier_scaled = scale_complex_by_magnitude(fourier_complex, shared_scale=True)

    fourier_phase_weight, fourier_phase_weight_scale = _magnitude_weight(
        fourier_scaled,
        lambda_value=PHASE_WEIGHT_LAMBDA,
        gamma=PHASE_WEIGHT_GAMMA,
    )

    image_magnitude = np.abs(image_scaled)
    fourier_magnitude = np.abs(fourier_scaled)
    fourier_log_magnitude = np.log1p(fourier_magnitude)
    fourier_phase, fourier_phase_crop_bounds = _center_crop(
        np.angle(fourier_scaled), FOURIER_PHASE_CROP_FRACTION
    )
    fourier_phase_weight_crop, _ = _center_crop(
        fourier_phase_weight, FOURIER_PHASE_CROP_FRACTION
    )

    # One robust display scale per magnitude representation, shared by all
    # four coils.  The underlying model input remains the shared 99th-percentile
    # scaled complex array above; log1p is used only for Fourier visualization.
    image_vmax = float(np.percentile(image_magnitude, 99.5))
    fourier_log_vmax = float(np.percentile(fourier_log_magnitude, 99.5))
    image_vmax = max(image_vmax, 1e-6)
    fourier_log_vmax = max(fourier_log_vmax, 1e-6)

    panel_names: dict[str, str] = {}
    panel_rows: list[dict[str, object]] = []
    for coil in range(4):
        coil_number = coil + 1
        outputs = {
            f"image_magnitude_coil{coil_number}": {
                "array": image_magnitude[coil],
                "cmap": "gray",
                "vmin": 0.0,
                "vmax": image_vmax,
                "mask": None,
                "phase_weight": None,
                "representation": "image-domain",
                "quantity": "magnitude",
                "phase_mask_rule": "not applicable",
                "magnitude_display_transform": "none; shared robustly scaled magnitude",
                "fft_normalization": "not applicable",
            },
            f"image_phase_coil{coil_number}": {
                "array": np.angle(image_scaled[coil]),
                "cmap": "twilight",
                "vmin": -np.pi,
                "vmax": np.pi,
                "mask": image_support,
                "phase_weight": None,
                "representation": "image-domain",
                "quantity": "phase",
                "phase_mask_rule": "RSS >= 0.02 * percentile_99(RSS), as used by align_multicoil_phase",
                "phase_weight_rule": "none; near-full opacity within the existing foreground support",
                "phase_weight_lambda": "not applicable",
                "phase_weight_gamma": "not applicable",
                "magnitude_display_transform": "not applicable",
                "fft_normalization": "not applicable",
            },
            f"fourier_log_magnitude_coil{coil_number}": {
                "array": fourier_log_magnitude[coil],
                "cmap": "gray",
                "vmin": 0.0,
                "vmax": fourier_log_vmax,
                "mask": None,
                "phase_weight": None,
                "representation": "Fourier-domain",
                "quantity": "log magnitude",
                "phase_mask_rule": "not applicable",
                "magnitude_display_transform": "log1p(scaled magnitude); visualization only",
                "fft_normalization": "centered orthonormal FFT (norm='ortho')",
            },
            f"fourier_phase_coil{coil_number}": {
                "array": fourier_phase[coil],
                "cmap": "twilight",
                "vmin": -np.pi,
                "vmax": np.pi,
                "mask": None,
                "phase_weight": fourier_phase_weight_crop[coil],
                "representation": "Fourier-domain",
                "quantity": "phase",
                "phase_mask_rule": "none; continuous magnitude weighting only",
                "phase_weight_rule": "w = clip(log1p(lambda * abs(X)) / percentile_99(log1p(lambda * abs(X))), 0, 1) ** gamma",
                "magnitude_display_transform": "not applicable",
                "fft_normalization": "centered orthonormal FFT (norm='ortho')",
            },
        }
        for name, details in outputs.items():
            filename = f"{name}.png"
            _save_panel(
                details["array"],
                OUTPUT_DIR / filename,
                cmap=details["cmap"],
                vmin=details["vmin"],
                vmax=details["vmax"],
                mask=details["mask"],
                phase_weight=details["phase_weight"],
            )
            panel_names[name] = filename
            panel_rows.append(
                {
                    "sample_path": str(selected_row["path"]),
                    "split": str(selected_row["data_split"]).strip().lower(),
                    "label": int(selected_row["label"]),
                    "coil": coil_number,
                    "representation": details["representation"],
                    "quantity": details["quantity"],
                    "panel_file": filename,
                    "phase_mask_rule": details["phase_mask_rule"],
                    "magnitude_display_transform": details["magnitude_display_transform"],
                    "fft_normalization": details["fft_normalization"],
                    "phase_weight_rule": details.get(
                        "phase_weight_rule",
                        "w = clip(log1p(lambda * abs(X)) / percentile_99(log1p(lambda * abs(X))), 0, 1) ** gamma",
                    ),
                    "phase_weight_lambda": details.get(
                        "phase_weight_lambda",
                        PHASE_WEIGHT_LAMBDA if details["quantity"] == "phase" else "not applicable",
                    ),
                    "phase_weight_gamma": details.get(
                        "phase_weight_gamma",
                        PHASE_WEIGHT_GAMMA if details["quantity"] == "phase" else "not applicable",
                    ),
                    "fourier_phase_crop": (
                        f"centered {FOURIER_PHASE_CROP_FRACTION:.2f} crop; bounds={fourier_phase_crop_bounds}"
                        if name.startswith("fourier_phase_")
                        else "not applicable"
                    ),
                    "scaling_rule": "scale_complex_by_magnitude(percentile=99.0, eps=1e-6, shared_scale=True)",
                    "phase_range": "[-pi, pi]" if details["quantity"] == "phase" else "not applicable",
                    "model_input_changed": False,
                }
            )

    row_data = {
        str(key): (None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value)
        for key, value in selected_row.items()
    }
    metadata = {
        "selected_manifest": str(manifest_path),
        "selection_rule": "lexicographically smallest path among data_split == training rows; stable tie-breakers fastmri_pt_id then slice_index",
        "selected_sample_path": str(sample_path),
        "selected_manifest_row": row_data,
        "input_key": "image_complex",
        "input_shape": list(raw_image.shape),
        "preprocessing": [
            "load image_complex",
            "verify exactly four complex coils",
            "align_multicoil_phase(image_complex)",
            "image representation: aligned complex image",
            "Fourier representation: centered_fft2(aligned_image)",
            "scale each representation with scale_complex_by_magnitude(..., shared_scale=True, percentile=99.0)",
        ],
        "phase_visualization": {
            "phase_hue": "angle of the scaled complex representation mapped with the cyclic twilight colormap over [-pi, pi]",
            "Fourier_phase_magnitude_weight": "w = clip(log1p(lambda * abs(X)) / percentile_99(log1p(lambda * abs(X))), 0, 1) ** gamma",
            "Fourier_phase_lambda": PHASE_WEIGHT_LAMBDA,
            "Fourier_phase_gamma": PHASE_WEIGHT_GAMMA,
            "Fourier_phase_weight_scope": "one percentile-normalized weight scale per Fourier representation across all four coils",
            "image_phase_opacity": "near-full within the existing foreground support mask; no magnitude weighting",
            "neutral_background": "white",
            "model_input_changed": False,
        },
        "image_phase_display": {
            "foreground_support": "RSS >= 0.02 * 99th percentile of RSS, matching align_multicoil_phase",
            "opacity": "near-full within the existing foreground support",
            "magnitude_weighting": "not applied",
            "model_input_changed": False,
        },
        "fourier_phase_display": {
            "weight_formula": "w = clip(log1p(lambda * abs(X)) / percentile_99(log1p(lambda * abs(X))), 0, 1) ** gamma",
            "lambda": PHASE_WEIGHT_LAMBDA,
            "gamma": PHASE_WEIGHT_GAMMA,
            "X_definition": "shared-robust-scaled centered orthonormal FFT representation",
            "weight_scale": fourier_phase_weight_scale,
            "central_crop_fraction": FOURIER_PHASE_CROP_FRACTION,
            "central_crop_bounds_y0_y1_x0_x1": list(fourier_phase_crop_bounds),
            "central_crop_shape": list(fourier_phase.shape[-2:]),
            "neutral_background": "white",
            "applies_to": "displayed Fourier-phase panels only",
            "model_input_changed": False,
        },
        "fourier_magnitude_display": "full-frame log1p of scaled magnitude for visualization only; model input remains the scaled complex Fourier representation",
        "image_support_pixels": int(image_support.sum()),
        "image_display_vmax": image_vmax,
        "fourier_log_display_vmax": fourier_log_vmax,
        "phase_range": [-float(np.pi), float(np.pi)],
        "panel_images": panel_names,
    }
    (OUTPUT_DIR / "figure1_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_fields = [
        "sample_path",
        "split",
        "label",
        "coil",
        "representation",
        "quantity",
        "panel_file",
        "phase_mask_rule",
        "phase_weight_rule",
        "phase_weight_lambda",
        "phase_weight_gamma",
        "fourier_phase_crop",
        "magnitude_display_transform",
        "fft_normalization",
        "scaling_rule",
        "phase_range",
        "model_input_changed",
    ]
    with (OUTPUT_DIR / "figure1_metadata.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(panel_rows)

    print(f"Selected manifest row: {row_data}")
    print(f"Selected sample path: {sample_path}")
    print(f"Wrote 16 high-resolution raster panels to {OUTPUT_DIR}")
    print(f"Image display vmax (shared across coils): {image_vmax:.6g}")
    print(f"Fourier log-magnitude display vmax (shared across coils): {fourier_log_vmax:.6g}")
    print(f"Phase weighting: lambda={PHASE_WEIGHT_LAMBDA:g}, gamma={PHASE_WEIGHT_GAMMA:g}")
    print(f"Fourier phase crop: centered {FOURIER_PHASE_CROP_FRACTION:.2f} ({fourier_phase.shape[-2]}x{fourier_phase.shape[-1]})")
    print("Fourier phase model input changed: False")


if __name__ == "__main__":
    main()
