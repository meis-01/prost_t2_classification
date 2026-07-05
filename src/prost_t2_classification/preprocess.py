from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from .image_ops import (
    center_crop_last2,
    middle_acquisition_index,
    pad_coil_axis,
    top_energy_coils,
)
from .labels import (
    load_t2_labels,
    normalize_exam_key,
    resolve_reconstruction_path,
    select_preprocessing_labels,
)
from .logging_utils import get_logger


@dataclass(frozen=True)
class ReconstructionJob:
    input_path: Path
    output_path: Path
    slice_one_based: Optional[int] = None


@dataclass(frozen=True)
class ReconstructionResult:
    source_path: Path
    output_path: Path
    original_shape: Tuple[int, ...]
    kspace_grappa_shape: Tuple[int, ...]
    image_complex_shape: Tuple[int, ...]
    acquisition_index: int
    slice_index: int


def iter_t2_h5_files(raw_root: Path) -> List[Path]:
    return sorted(
        path
        for path in raw_root.rglob("file_prostate_AXT2_*.h5")
        if "_complex_recon" not in path.name
    )


def build_reconstruction_jobs(
    raw_root: Path,
    recon_root: Path,
    selected_labels=None,
) -> List[ReconstructionJob]:
    selection_by_key: Dict[tuple[str, str], int] = {}
    selection_by_rawfile: Dict[str, int] = {}
    if selected_labels is not None:
        for row in selected_labels[["folder", "fastmri_rawfile", "slice"]].itertuples(index=False):
            key = normalize_exam_key(str(row.folder), str(row.fastmri_rawfile))
            slice_one_based = int(row.slice)
            selection_by_key[key] = slice_one_based
            selection_by_rawfile[key[1]] = slice_one_based

    jobs: List[ReconstructionJob] = []
    for input_path in iter_t2_h5_files(raw_root):
        relative_parent = input_path.parent.relative_to(raw_root)
        output_path = recon_root / relative_parent / f"{input_path.stem}_complex_recon.h5"
        slice_one_based = None
        if selected_labels is not None:
            key = normalize_exam_key(relative_parent.as_posix(), input_path.name)
            slice_one_based = selection_by_key.get(key, selection_by_rawfile.get(input_path.name))
            if slice_one_based is None:
                continue
        jobs.append(
            ReconstructionJob(
                input_path=input_path,
                output_path=output_path,
                slice_one_based=slice_one_based,
            )
        )
    return jobs


def reconstruct_t2_dataset(
    raw_root: Path,
    recon_root: Path,
    *,
    kernel_size: Tuple[int, int] = (5, 5),
    skip_existing: bool = True,
    limit: Optional[int] = None,
    selected_exams: Optional[Iterable[tuple[str, str]]] = None,
    selected_labels=None,
) -> List[Path]:
    logger = get_logger()
    try:
        from fastmri_tools.prostate_opts.fft import centered_ifft
        from fastmri_tools.prostate_opts.grappa import grappa_fill
    except ImportError as exc:
        raise RuntimeError(
            "fastmri-tools is required for reconstruction. Install with `python -m pip install fastmri-tools`."
        ) from exc

    jobs = build_reconstruction_jobs(raw_root, recon_root, selected_labels=selected_labels)
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
    logger.info("Starting selected T2 reconstruction for %d files", len(jobs))
    for job in tqdm(jobs, desc="reconstruct T2"):
        outputs.append(job.output_path)
        if job.output_path.exists() and skip_existing:
            logger.info("Skipping existing reconstruction %s", job.output_path)
            continue
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        result = reconstruct_selected_t2_file(
            job.input_path,
            job.output_path,
            centered_ifft=centered_ifft,
            grappa_fill=grappa_fill,
            kernel_size=kernel_size,
            slice_one_based=job.slice_one_based,
        )
        logger.info(
            "Reconstructed %s -> %s acquisition=%d slice=%d image_shape=%s",
            result.source_path,
            result.output_path,
            result.acquisition_index,
            result.slice_index + 1,
            result.image_complex_shape,
        )
    return outputs


def reconstruct_selected_t2_file(
    input_path: Path,
    output_path: Path,
    *,
    centered_ifft,
    grappa_fill,
    kernel_size: Tuple[int, int],
    slice_one_based: Optional[int],
) -> ReconstructionResult:
    with h5py.File(input_path, "r") as h5:
        if "kspace" not in h5:
            raise KeyError(f"{input_path} does not contain a 'kspace' dataset.")
        if "calibration_data" not in h5:
            raise KeyError(f"{input_path} does not contain a 'calibration_data' dataset.")

        kspace_dataset = h5["kspace"]
        calibration_dataset = h5["calibration_data"]
        if kspace_dataset.ndim != 5:
            raise ValueError(
                f"{input_path} kspace must have shape (averages, slices, coils, readout, phase); "
                f"got {kspace_dataset.shape}."
            )
        if calibration_dataset.ndim != 4:
            raise ValueError(
                f"{input_path} calibration_data must have shape (slices, coils, readout, calibration_phase); "
                f"got {calibration_dataset.shape}."
            )

        original_shape = tuple(kspace_dataset.shape)
        acquisition_index = middle_acquisition_index(original_shape[0])
        slice_index = slice_one_based - 1 if slice_one_based is not None else original_shape[1] // 2
        if slice_index < 0 or slice_index >= original_shape[1]:
            raise ValueError(
                f"Selected slice {slice_index + 1} is out of range for {input_path} with shape {original_shape}."
            )

        kspace = _as_complex(kspace_dataset[acquisition_index : acquisition_index + 1, slice_index : slice_index + 1])
        calibration = _as_complex(calibration_dataset[slice_index : slice_index + 1])
        source_attrs = {key: _attr_value(value) for key, value in h5.attrs.items()}

    kspace_grappa = grappa_fill(kspace, calibration, kernel_size=kernel_size)
    image_complex = centered_ifft(kspace_grappa, axes=(-2, -1))

    write_selected_reconstruction(
        output_path,
        source_path=input_path,
        source_attrs=source_attrs,
        original_shape=original_shape,
        kspace_regridded=kspace,
        kspace_grappa=kspace_grappa,
        image_complex=image_complex,
        acquisition_index=acquisition_index,
        slice_index=slice_index,
    )

    return ReconstructionResult(
        source_path=Path(input_path),
        output_path=Path(output_path),
        original_shape=original_shape,
        kspace_grappa_shape=tuple(kspace_grappa.shape),
        image_complex_shape=tuple(image_complex.shape),
        acquisition_index=acquisition_index,
        slice_index=slice_index,
    )


def write_selected_reconstruction(
    output_path: Path,
    *,
    source_path: Path,
    source_attrs: Mapping[str, Any],
    original_shape: Tuple[int, ...],
    kspace_regridded: np.ndarray,
    kspace_grappa: np.ndarray,
    image_complex: np.ndarray,
    acquisition_index: int,
    slice_index: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("kspace_regridded", data=kspace_regridded)
        h5.create_dataset("kspace_grappa", data=kspace_grappa)
        h5.create_dataset("image_complex", data=image_complex)
        h5.attrs["source_file"] = str(source_path)
        h5.attrs["sequence"] = "t2"
        h5.attrs["original_kspace_shape"] = ",".join(map(str, original_shape))
        h5.attrs["complex_output"] = True
        h5.attrs["spatial_fft_axes"] = "readout,phase"
        h5.attrs["subset_reconstruction"] = True
        h5.attrs["selected_acquisition_index"] = acquisition_index
        h5.attrs["selected_slice_index"] = slice_index
        h5.attrs["selected_slice"] = slice_index + 1

        source_attrs_group = h5.create_group("source_attrs")
        for key, value in source_attrs.items():
            if _is_hdf5_attr_value(value):
                source_attrs_group.attrs[key] = value
            else:
                source_attrs_group.attrs[key] = str(value)


def _as_complex(array: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(array):
        return np.asarray(array)
    return np.asarray(array).astype(np.complex64)


def _attr_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_hdf5_attr_value(value: Any) -> bool:
    return isinstance(value, (str, bytes, int, float, bool, np.number, np.ndarray))


def read_selected_reconstruction_slice(
    h5: h5py.File,
    recon_path: Path,
    *,
    requested_slice_index: int,
) -> tuple[np.ndarray, int]:
    image_dataset = h5["image_complex"]
    is_subset = bool(h5.attrs.get("subset_reconstruction", False)) or image_dataset.shape[:2] == (1, 1)

    if is_subset:
        stored_slice_index = int(h5.attrs.get("selected_slice_index", requested_slice_index))
        if stored_slice_index != requested_slice_index:
            raise ValueError(
                f"{recon_path} contains selected slice {stored_slice_index + 1}, but labels request "
                f"slice {requested_slice_index + 1}. Re-run reconstruction with the same labels."
            )
        acquisition_index = int(h5.attrs.get("selected_acquisition_index", 0))
        return np.asarray(image_dataset[0, 0]), acquisition_index

    acquisition_index = middle_acquisition_index(image_dataset.shape[0])
    if requested_slice_index < 0 or requested_slice_index >= image_dataset.shape[1]:
        raise ValueError(
            f"Selected slice {requested_slice_index + 1} is out of range for {recon_path} "
            f"with shape {image_dataset.shape}."
        )
    return np.asarray(image_dataset[acquisition_index, requested_slice_index]), acquisition_index


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
    labels = select_preprocessing_labels(
        labels,
        split_exam_counts=split_exam_counts,
        limit_patients=limit_patients,
        limit_slices=limit_slices,
    )

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

            for _, row in group.iterrows():
                slice_one_based = int(row["slice"])
                slice_index = slice_one_based - 1
                all_coils_slice, acquisition_index = read_selected_reconstruction_slice(
                    h5,
                    recon_path,
                    requested_slice_index=slice_index,
                )
                coil_energy = np.sum(np.abs(all_coils_slice) ** 2, axis=(-2, -1))
                selected_coils = top_energy_coils(coil_energy, max_coils=max_coils)
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
