from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from .image_ops import (
    center_crop_last2,
    middle_acquisition_index,
    pad_coil_axis,
    top_energy_coils_from_stream,
)
from .labels import (
    assert_patient_split_disjoint,
    load_t2_labels,
    normalize_exam_key,
    resolve_reconstruction_path,
    select_split_exams,
)
from .logging_utils import get_logger


@dataclass(frozen=True)
class ReconstructionJob:
    input_path: Path
    output_path: Path


def iter_t2_h5_files(raw_root: Path) -> List[Path]:
    return sorted(
        path
        for path in raw_root.rglob("file_prostate_AXT2_*.h5")
        if "_complex_recon" not in path.name
    )


def build_reconstruction_jobs(raw_root: Path, recon_root: Path) -> List[ReconstructionJob]:
    jobs: List[ReconstructionJob] = []
    for input_path in iter_t2_h5_files(raw_root):
        relative_parent = input_path.parent.relative_to(raw_root)
        output_path = recon_root / relative_parent / f"{input_path.stem}_complex_recon.h5"
        jobs.append(ReconstructionJob(input_path=input_path, output_path=output_path))
    return jobs


def reconstruct_t2_dataset(
    raw_root: Path,
    recon_root: Path,
    *,
    kernel_size: Tuple[int, int] = (5, 5),
    skip_existing: bool = True,
    limit: Optional[int] = None,
    selected_exams: Optional[Iterable[tuple[str, str]]] = None,
) -> List[Path]:
    logger = get_logger()
    try:
        from fastmri_tools.prostate_opts.pipeline import reconstruct_file
    except ImportError as exc:
        raise RuntimeError(
            "fastmri-tools is required for reconstruction. Install with `python -m pip install fastmri-tools`."
        ) from exc

    jobs = build_reconstruction_jobs(raw_root, recon_root)
    if selected_exams is not None:
        selected = {normalize_exam_key(folder, rawfile) for folder, rawfile in selected_exams}
        selected_rawfiles = {rawfile for _, rawfile in selected}
        jobs = [
            job
            for job in jobs
            if (
                job_key := normalize_exam_key(
                    job.input_path.parent.relative_to(raw_root).as_posix(),
                    job.input_path.name,
                )
            )
            in selected
            or job_key[1] in selected_rawfiles
        ]
    if limit is not None:
        jobs = jobs[:limit]
    if not jobs:
        raise FileNotFoundError(f"No T2 H5 files found under {raw_root}.")

    outputs: List[Path] = []
    logger.info("Starting T2 reconstruction for %d files", len(jobs))
    for job in tqdm(jobs, desc="reconstruct T2"):
        outputs.append(job.output_path)
        if job.output_path.exists() and skip_existing:
            logger.info("Skipping existing reconstruction %s", job.output_path)
            continue
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        result = reconstruct_file(
            job.input_path,
            job.output_path,
            sequence="t2",
            kernel_size=kernel_size,
        )
        logger.info(
            "Reconstructed %s -> %s image_shape=%s",
            result.source_path,
            result.output_path,
            result.image_complex_shape,
        )
    return outputs


def make_npz_dataset(
    labels_path: Path,
    recon_root: Path,
    npz_root: Path,
    *,
    crop_size: int = 224,
    max_coils: int = 5,
    overwrite: bool = False,
    limit_patients: Optional[int] = None,
    limit_slices: Optional[int] = None,
    split_exam_counts: Optional[Mapping[str, int]] = None,
) -> Path:
    logger = get_logger()
    labels = load_t2_labels(labels_path)
    assert_patient_split_disjoint(labels)

    if split_exam_counts is not None:
        labels = select_split_exams(labels, split_exam_counts)
    if limit_patients is not None:
        keep_patients = sorted(labels["fastmri_pt_id"].unique())[:limit_patients]
        labels = labels[labels["fastmri_pt_id"].isin(keep_patients)].copy()
    if limit_slices is not None:
        labels = labels.head(limit_slices).copy()

    npz_root.mkdir(parents=True, exist_ok=True)
    samples_dir = npz_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = npz_root / "manifest.csv"

    rows: List[Dict[str, object]] = []
    grouped = labels.groupby(["folder", "fastmri_rawfile"], sort=True)
    logger.info("Creating NPZ samples for %d reconstructed T2 volumes", len(grouped))

    for (folder, rawfile), group in tqdm(grouped, desc="make NPZ"):
        recon_path = resolve_reconstruction_path(recon_root, str(folder), str(rawfile))
        with h5py.File(recon_path, "r") as h5:
            if "image_complex" not in h5:
                raise KeyError(f"{recon_path} does not contain image_complex.")
            image_dataset = h5["image_complex"]
            if image_dataset.ndim != 5:
                raise ValueError(f"{recon_path} image_complex has unexpected shape {image_dataset.shape}")

            acquisition_index = middle_acquisition_index(image_dataset.shape[0])
            selected_coils = top_energy_coils_from_stream(
                image_dataset,
                acquisition_index=acquisition_index,
                max_coils=max_coils,
            )

            for _, row in group.iterrows():
                slice_one_based = int(row["slice"])
                slice_index = slice_one_based - 1
                if slice_index < 0 or slice_index >= image_dataset.shape[1]:
                    logger.warning(
                        "Skipping out-of-range slice %s for %s with shape %s",
                        slice_one_based,
                        recon_path,
                        image_dataset.shape,
                    )
                    continue

                all_coils_slice = np.asarray(image_dataset[acquisition_index, slice_index])
                selected = all_coils_slice[selected_coils]
                selected = center_crop_last2(selected, crop_size)
                selected = pad_coil_axis(selected.astype(np.complex64), max_coils)

                patient_id = int(row["fastmri_pt_id"])
                sample_name = f"pt{patient_id:03d}_slice{slice_one_based:03d}.npz"
                sample_path = samples_dir / sample_name
                if sample_path.exists() and not overwrite:
                    pass
                else:
                    np.savez_compressed(
                        sample_path,
                        image_complex=selected,
                        patient_id=np.int32(patient_id),
                        slice=np.int32(slice_one_based),
                        pirads=np.int32(row["PIRADS"]),
                        label=np.int32(row["label"]),
                        split=str(row["data_split"]),
                        acquisition_index=np.int32(acquisition_index),
                        selected_coils=selected_coils.astype(np.int32),
                    )

                rows.append(
                    {
                        "path": sample_path.relative_to(npz_root).as_posix(),
                        "fastmri_pt_id": patient_id,
                        "slice": slice_one_based,
                        "slice_index": slice_index,
                        "PIRADS": int(row["PIRADS"]),
                        "label": int(row["label"]),
                        "data_split": str(row["data_split"]),
                        "folder": str(folder),
                        "fastmri_rawfile": str(rawfile),
                        "source_recon": str(recon_path),
                        "acquisition_index": acquisition_index,
                        "selected_coils": ";".join(str(int(coil)) for coil in selected_coils),
                        "channels": int(selected.shape[0]),
                        "height": int(selected.shape[-2]),
                        "width": int(selected.shape[-1]),
                    }
                )

    write_manifest(manifest_path, rows)
    logger.info("Wrote manifest with %d samples to %s", len(rows), manifest_path)
    return manifest_path


def write_manifest(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No NPZ samples were written; manifest would be empty.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
