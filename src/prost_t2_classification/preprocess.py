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
    middle_coil_index,
    pad_coil_axis,
)
from .labels import (
    load_t2_labels,
    normalize_exam_key,
    resolve_reconstruction_path,
    select_preprocessing_labels,
)
from .logging_utils import get_logger


SKIPPABLE_DATA_ERRORS = (OSError, RuntimeError, KeyError, ValueError)


@dataclass(frozen=True)
class ReconstructionJob:
    input_path: Path
    output_path: Path
    slice_numbers_one_based: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class ReconstructionResult:
    source_path: Path
    output_path: Path
    original_shape: Tuple[int, ...]
    image_complex_shape: Tuple[int, ...]
    acquisition_index: int
    slice_indices: Tuple[int, ...]
    coil_indices: Tuple[int, ...]


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
    selection_by_key: Dict[tuple[str, str], List[int]] = {}
    selection_by_rawfile: Dict[str, List[int]] = {}
    if selected_labels is not None:
        for row in selected_labels[["folder", "fastmri_rawfile", "slice"]].itertuples(index=False):
            key = normalize_exam_key(str(row.folder), str(row.fastmri_rawfile))
            slice_one_based = int(row.slice)
            selection_by_key.setdefault(key, []).append(slice_one_based)
            selection_by_rawfile.setdefault(key[1], []).append(slice_one_based)

    jobs: List[ReconstructionJob] = []
    for input_path in iter_t2_h5_files(raw_root):
        relative_parent = input_path.parent.relative_to(raw_root)
        output_path = recon_root / relative_parent / f"{input_path.stem}_complex_recon.h5"
        slice_numbers_one_based = None
        if selected_labels is not None:
            key = normalize_exam_key(relative_parent.as_posix(), input_path.name)
            selected_slices = selection_by_key.get(key, selection_by_rawfile.get(input_path.name))
            if selected_slices is None:
                continue
            slice_numbers_one_based = tuple(sorted(set(selected_slices)))
        jobs.append(
            ReconstructionJob(
                input_path=input_path,
                output_path=output_path,
                slice_numbers_one_based=slice_numbers_one_based,
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
    if limit is not None:
        jobs = jobs[:limit]
    if not jobs:
        raise FileNotFoundError(f"No T2 H5 files found under {raw_root}.")

    outputs: List[Path] = []
    failures: List[Dict[str, object]] = []
    failure_path = recon_root / "failed_reconstructions.csv"
    logger.info("Starting selected T2 reconstruction for %d files", len(jobs))
    for job in tqdm(jobs, desc="reconstruct T2"):
        if job.output_path.exists() and skip_existing:
            outputs.append(job.output_path)
            logger.info("Skipping existing reconstruction %s", job.output_path)
            continue
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = reconstruct_selected_t2_file(
                job.input_path,
                job.output_path,
                centered_ifft=centered_ifft,
                grappa_fill=grappa_fill,
                kernel_size=kernel_size,
                slice_numbers_one_based=job.slice_numbers_one_based,
            )
        except SKIPPABLE_DATA_ERRORS as exc:
            job.output_path.unlink(missing_ok=True)
            failures.append(
                {
                    "input_path": str(job.input_path),
                    "output_path": str(job.output_path),
                    "slices": ";".join(map(str, job.slice_numbers_one_based or ())),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            logger.warning(
                "Skipping reconstruction for %s: %s: %s",
                job.input_path,
                type(exc).__name__,
                exc,
            )
            continue
        outputs.append(job.output_path)
        logger.info(
            "Reconstructed %s -> %s acquisition=%d slice_count=%d image_shape=%s",
            result.source_path,
            result.output_path,
            result.acquisition_index,
            len(result.slice_indices),
            result.image_complex_shape,
        )
    write_optional_csv(failure_path, failures)
    if failures:
        logger.warning("Skipped %d reconstruction file(s); details written to %s", len(failures), failure_path)
    if not outputs:
        raise RuntimeError(f"All reconstruction jobs failed; see {failure_path}.")
    return outputs


def reconstruct_selected_t2_file(
    input_path: Path,
    output_path: Path,
    *,
    centered_ifft,
    grappa_fill,
    kernel_size: Tuple[int, int],
    slice_numbers_one_based: Optional[Iterable[int]],
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
        coil_indices = (middle_coil_index(original_shape[2]),)
        slice_indices = selected_slice_indices(slice_numbers_one_based, original_shape[1], input_path)
        source_attrs = {key: _attr_value(value) for key, value in h5.attrs.items()}

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_complex_shape: Tuple[int, ...] | None = None
        with h5py.File(output_path, "w") as output_h5:
            write_reconstruction_metadata(
                output_h5,
                source_path=input_path,
                source_attrs=source_attrs,
                original_shape=original_shape,
                acquisition_index=acquisition_index,
                slice_indices=slice_indices,
                coil_indices=coil_indices,
            )
            image_dataset = None
            for output_slice_index, source_slice_index in enumerate(slice_indices):
                kspace = _as_complex(
                    kspace_dataset[
                        acquisition_index : acquisition_index + 1,
                        source_slice_index : source_slice_index + 1,
                    ]
                )
                calibration = _as_complex(calibration_dataset[source_slice_index : source_slice_index + 1])
                image_complex = centered_ifft(
                    grappa_fill(kspace, calibration, kernel_size=kernel_size),
                    axes=(-2, -1),
                ).astype(np.complex64, copy=False)
                image_complex = image_complex[:, :, coil_indices[0] : coil_indices[0] + 1]

                if image_dataset is None:
                    image_complex_shape = (1, len(slice_indices), *image_complex.shape[2:])
                    image_dataset = output_h5.create_dataset(
                        "image_complex",
                        shape=image_complex_shape,
                        dtype=image_complex.dtype,
                    )
                image_dataset[:, output_slice_index : output_slice_index + 1] = image_complex

    if image_complex_shape is None:
        raise ValueError(f"No slices selected for {input_path}.")

    return ReconstructionResult(
        source_path=Path(input_path),
        output_path=Path(output_path),
        original_shape=original_shape,
        image_complex_shape=image_complex_shape,
        acquisition_index=acquisition_index,
        slice_indices=slice_indices,
        coil_indices=coil_indices,
    )


def selected_slice_indices(
    slice_numbers_one_based: Optional[Iterable[int]],
    num_slices: int,
    input_path: Path,
) -> Tuple[int, ...]:
    if num_slices <= 0:
        raise ValueError(f"{input_path} has no slices.")
    if slice_numbers_one_based is None:
        return tuple(range(num_slices))

    indices = tuple(sorted({int(slice_number) - 1 for slice_number in slice_numbers_one_based}))
    if not indices:
        raise ValueError(f"No slices selected for {input_path}.")
    invalid = [index + 1 for index in indices if index < 0 or index >= num_slices]
    if invalid:
        raise ValueError(
            f"Selected slice(s) {invalid} are out of range for {input_path} with {num_slices} slices."
        )
    return indices


def write_reconstruction_metadata(
    h5: h5py.File,
    *,
    source_path: Path,
    source_attrs: Mapping[str, Any],
    original_shape: Tuple[int, ...],
    acquisition_index: int,
    slice_indices: Tuple[int, ...],
    coil_indices: Tuple[int, ...],
) -> None:
    h5.attrs["source_file"] = str(source_path)
    h5.attrs["sequence"] = "t2"
    h5.attrs["original_kspace_shape"] = ",".join(map(str, original_shape))
    h5.attrs["complex_output"] = True
    h5.attrs["spatial_fft_axes"] = "readout,phase"
    h5.attrs["subset_reconstruction"] = True
    h5.attrs["selected_acquisition_index"] = acquisition_index
    h5.attrs["selected_slice_indices"] = np.asarray(slice_indices, dtype=np.int32)
    h5.attrs["selected_slices"] = np.asarray([index + 1 for index in slice_indices], dtype=np.int32)
    h5.attrs["selected_coil_indices"] = np.asarray(coil_indices, dtype=np.int32)
    h5.attrs["selected_coils"] = np.asarray([index + 1 for index in coil_indices], dtype=np.int32)

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
    if image_dataset.ndim != 5 or image_dataset.shape[0] != 1:
        raise ValueError(
            f"{recon_path} must be rebuilt with selected-acquisition reconstruction; "
            f"found image_complex shape {image_dataset.shape}."
        )
    if "selected_slice_indices" not in h5.attrs or "selected_acquisition_index" not in h5.attrs:
        raise ValueError(f"{recon_path} is missing selected reconstruction metadata.")

    selected_indices = tuple(int(index) for index in np.asarray(h5.attrs["selected_slice_indices"]).reshape(-1))
    try:
        output_slice_index = selected_indices.index(requested_slice_index)
    except ValueError as exc:
        raise ValueError(
            f"{recon_path} does not contain requested slice {requested_slice_index + 1}. "
            "Re-run reconstruction with the same labels."
        ) from exc
    if output_slice_index >= image_dataset.shape[1]:
        raise ValueError(
            f"{recon_path} selected-slice metadata does not match image_complex shape {image_dataset.shape}."
        )
    acquisition_index = int(h5.attrs["selected_acquisition_index"])
    return np.asarray(image_dataset[0, output_slice_index]), acquisition_index


def reconstruction_coil_indices(h5: h5py.File, num_coils: int) -> tuple[np.ndarray, bool]:
    if "selected_coil_indices" in h5.attrs:
        indices = np.asarray(h5.attrs["selected_coil_indices"], dtype=np.int64).reshape(-1)
        if len(indices) == num_coils:
            return indices, True
    return np.arange(num_coils, dtype=np.int64), False


def make_npz_dataset(
    labels_path: Path,
    recon_root: Path,
    npz_root: Path,
    *,
    crop_size: int = 224,
    max_coils: int = 1,
    overwrite: bool = False,
    limit_patients: Optional[int] = None,
    limit_slices: Optional[int] = None,
    split_exam_counts: Optional[Mapping[str, int]] = None,
) -> Path:
    logger = get_logger()
    if max_coils <= 0:
        raise ValueError("max_coils must be positive.")
    labels = load_t2_labels(labels_path)
    labels = select_preprocessing_labels(
        labels,
        split_exam_counts=split_exam_counts,
        limit_patients=limit_patients,
        limit_slices=limit_slices,
        middle_slice_only=split_exam_counts is not None,
    )

    npz_root.mkdir(parents=True, exist_ok=True)
    samples_dir = npz_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = npz_root / "manifest.csv"

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    failure_path = npz_root / "failed_npz.csv"
    grouped = labels.groupby(["folder", "fastmri_rawfile"], sort=True)
    logger.info("Creating NPZ samples for %d reconstructed T2 volumes", len(grouped))

    for (folder, rawfile), group in tqdm(grouped, desc="make NPZ"):
        try:
            recon_path = resolve_reconstruction_path(recon_root, str(folder), str(rawfile))
            prepared_rows = prepare_npz_rows_for_exam(
                group,
                recon_path,
                samples_dir=samples_dir,
                crop_size=crop_size,
                max_coils=max_coils,
            )
        except SKIPPABLE_DATA_ERRORS as exc:
            failures.append(
                {
                    "folder": str(folder),
                    "fastmri_rawfile": str(rawfile),
                    "recon_path": str(recon_root / str(folder) / f"{Path(str(rawfile)).stem}_complex_recon.h5"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            logger.warning(
                "Skipping NPZ creation for %s/%s: %s: %s",
                folder,
                rawfile,
                type(exc).__name__,
                exc,
            )
            continue

        for prepared in prepared_rows:
            sample_path = prepared["sample_path"]
            if sample_path.exists() and not overwrite:
                pass
            else:
                np.savez_compressed(
                    sample_path,
                    image_complex=prepared["image_complex"],
                    patient_id=np.int32(prepared["fastmri_pt_id"]),
                    slice=np.int32(prepared["slice"]),
                    pirads=np.int32(prepared["PIRADS"]),
                    label=np.int32(prepared["label"]),
                    split=str(prepared["data_split"]),
                    acquisition_index=np.int32(prepared["acquisition_index"]),
                    selected_coils=prepared["selected_coils"].astype(np.int32),
                )

            rows.append(
                {
                    "path": sample_path.relative_to(npz_root).as_posix(),
                    "fastmri_pt_id": int(prepared["fastmri_pt_id"]),
                    "slice": int(prepared["slice"]),
                    "slice_index": int(prepared["slice_index"]),
                    "PIRADS": int(prepared["PIRADS"]),
                    "label": int(prepared["label"]),
                    "data_split": str(prepared["data_split"]),
                    "folder": str(prepared["folder"]),
                    "fastmri_rawfile": str(prepared["fastmri_rawfile"]),
                    "source_recon": str(prepared["source_recon"]),
                    "acquisition_index": int(prepared["acquisition_index"]),
                    "selected_coils": ";".join(str(int(coil)) for coil in prepared["selected_coils"]),
                    "channels": int(prepared["image_complex"].shape[0]),
                    "height": int(prepared["image_complex"].shape[-2]),
                    "width": int(prepared["image_complex"].shape[-1]),
                }
            )

    write_optional_csv(failure_path, failures)
    if failures:
        logger.warning("Skipped %d NPZ exam(s); details written to %s", len(failures), failure_path)
    write_manifest(manifest_path, rows)
    logger.info("Wrote manifest with %d samples to %s", len(rows), manifest_path)
    return manifest_path


def make_kspace_npz_dataset(
    labels_path: Path,
    raw_root: Path,
    npz_root: Path,
    *,
    crop_size: int = 224,
    max_coils: int = 1,
    overwrite: bool = False,
    limit_patients: Optional[int] = None,
    limit_slices: Optional[int] = None,
    split_exam_counts: Optional[Mapping[str, int]] = None,
) -> Path:
    logger = get_logger()
    if max_coils <= 0:
        raise ValueError("max_coils must be positive.")
    labels = load_t2_labels(labels_path)
    labels = select_preprocessing_labels(
        labels,
        split_exam_counts=split_exam_counts,
        limit_patients=limit_patients,
        limit_slices=limit_slices,
        middle_slice_only=split_exam_counts is not None,
    )

    npz_root.mkdir(parents=True, exist_ok=True)
    samples_dir = npz_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = npz_root / "manifest.csv"

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    failure_path = npz_root / "failed_npz.csv"
    grouped = labels.groupby(["folder", "fastmri_rawfile"], sort=True)
    logger.info("Creating k-space NPZ samples for %d T2 volumes", len(grouped))

    for (folder, rawfile), group in tqdm(grouped, desc="make k-space NPZ"):
        raw_path = raw_root / str(folder) / str(rawfile)
        try:
            prepared_rows = prepare_kspace_npz_rows_for_exam(
                group,
                raw_path,
                samples_dir=samples_dir,
                crop_size=crop_size,
                max_coils=max_coils,
            )
        except SKIPPABLE_DATA_ERRORS as exc:
            failures.append(
                {
                    "folder": str(folder),
                    "fastmri_rawfile": str(rawfile),
                    "raw_path": str(raw_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            logger.warning(
                "Skipping k-space NPZ creation for %s/%s: %s: %s",
                folder,
                rawfile,
                type(exc).__name__,
                exc,
            )
            continue

        for prepared in prepared_rows:
            sample_path = prepared["sample_path"]
            if sample_path.exists() and not overwrite:
                pass
            else:
                np.savez_compressed(
                    sample_path,
                    kspace_complex=prepared["kspace_complex"],
                    patient_id=np.int32(prepared["fastmri_pt_id"]),
                    slice=np.int32(prepared["slice"]),
                    pirads=np.int32(prepared["PIRADS"]),
                    label=np.int32(prepared["label"]),
                    split=str(prepared["data_split"]),
                    acquisition_index=np.int32(prepared["acquisition_index"]),
                    selected_coils=prepared["selected_coils"].astype(np.int32),
                )

            rows.append(
                {
                    "path": sample_path.relative_to(npz_root).as_posix(),
                    "fastmri_pt_id": int(prepared["fastmri_pt_id"]),
                    "slice": int(prepared["slice"]),
                    "slice_index": int(prepared["slice_index"]),
                    "PIRADS": int(prepared["PIRADS"]),
                    "label": int(prepared["label"]),
                    "data_split": str(prepared["data_split"]),
                    "folder": str(prepared["folder"]),
                    "fastmri_rawfile": str(prepared["fastmri_rawfile"]),
                    "source_kspace": str(prepared["source_kspace"]),
                    "acquisition_index": int(prepared["acquisition_index"]),
                    "selected_coils": ";".join(str(int(coil)) for coil in prepared["selected_coils"]),
                    "channels": int(prepared["kspace_complex"].shape[0]),
                    "height": int(prepared["kspace_complex"].shape[-2]),
                    "width": int(prepared["kspace_complex"].shape[-1]),
                }
            )

    write_optional_csv(failure_path, failures)
    if failures:
        logger.warning("Skipped %d k-space NPZ exam(s); details written to %s", len(failures), failure_path)
    write_manifest(manifest_path, rows)
    logger.info("Wrote k-space manifest with %d samples to %s", len(rows), manifest_path)
    return manifest_path


def prepare_kspace_npz_rows_for_exam(
    group,
    raw_path: Path,
    *,
    samples_dir: Path,
    crop_size: int,
    max_coils: int,
) -> List[Dict[str, object]]:
    if max_coils <= 0:
        raise ValueError("max_coils must be positive.")
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    prepared_rows: List[Dict[str, object]] = []
    with h5py.File(raw_path, "r") as h5:
        if "kspace" not in h5:
            raise KeyError(f"{raw_path} does not contain kspace.")
        kspace_dataset = h5["kspace"]
        if kspace_dataset.ndim != 5:
            raise ValueError(
                f"{raw_path} kspace must have shape (averages, slices, coils, readout, phase); "
                f"got {kspace_dataset.shape}."
            )
        acquisition_index = middle_acquisition_index(kspace_dataset.shape[0])
        selected_coils = np.asarray([middle_coil_index(kspace_dataset.shape[2])], dtype=np.int64)

        for _, row in group.iterrows():
            slice_one_based = int(row["slice"])
            slice_index = slice_one_based - 1
            if slice_index < 0 or slice_index >= kspace_dataset.shape[1]:
                raise ValueError(
                    f"Selected slice {slice_one_based} is out of range for {raw_path} "
                    f"with {kspace_dataset.shape[1]} slices."
                )
            kspace_complex = _as_complex(
                kspace_dataset[
                    acquisition_index,
                    slice_index,
                    selected_coils,
                ]
            ).astype(np.complex64, copy=False)
            kspace_complex = center_crop_last2(kspace_complex, crop_size)
            kspace_complex = pad_coil_axis(kspace_complex, max_coils)

            patient_id = int(row["fastmri_pt_id"])
            sample_name = f"pt{patient_id:03d}_slice{slice_one_based:03d}.npz"
            prepared_rows.append(
                {
                    "sample_path": samples_dir / sample_name,
                    "kspace_complex": kspace_complex,
                    "fastmri_pt_id": patient_id,
                    "slice": slice_one_based,
                    "slice_index": slice_index,
                    "PIRADS": int(row["PIRADS"]),
                    "label": int(row["label"]),
                    "data_split": str(row["data_split"]),
                    "folder": str(row["folder"]),
                    "fastmri_rawfile": str(row["fastmri_rawfile"]),
                    "source_kspace": raw_path,
                    "acquisition_index": acquisition_index,
                    "selected_coils": selected_coils,
                }
            )
    return prepared_rows


def prepare_npz_rows_for_exam(
    group,
    recon_path: Path,
    *,
    samples_dir: Path,
    crop_size: int,
    max_coils: int,
) -> List[Dict[str, object]]:
    if max_coils <= 0:
        raise ValueError("max_coils must be positive.")
    prepared_rows: List[Dict[str, object]] = []
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
            source_coils, already_selected = reconstruction_coil_indices(h5, all_coils_slice.shape[0])
            if already_selected:
                selected_coils = source_coils
                selected = all_coils_slice
            else:
                local_coil_indices = np.asarray([middle_coil_index(all_coils_slice.shape[0])], dtype=np.int64)
                selected_coils = source_coils[local_coil_indices]
                selected = all_coils_slice[local_coil_indices]
            selected = center_crop_last2(selected, crop_size)
            selected = pad_coil_axis(selected.astype(np.complex64), max_coils)

            patient_id = int(row["fastmri_pt_id"])
            sample_name = f"pt{patient_id:03d}_slice{slice_one_based:03d}.npz"
            prepared_rows.append(
                {
                    "sample_path": samples_dir / sample_name,
                    "image_complex": selected,
                    "fastmri_pt_id": patient_id,
                    "slice": slice_one_based,
                    "slice_index": slice_index,
                    "PIRADS": int(row["PIRADS"]),
                    "label": int(row["label"]),
                    "data_split": str(row["data_split"]),
                    "folder": str(row["folder"]),
                    "fastmri_rawfile": str(row["fastmri_rawfile"]),
                    "source_recon": recon_path,
                    "acquisition_index": acquisition_index,
                    "selected_coils": selected_coils,
                }
            )
    return prepared_rows


def write_manifest(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No NPZ samples were written; manifest would be empty.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_optional_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
