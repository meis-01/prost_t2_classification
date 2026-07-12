import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from prost_t2_classification.cli import build_parser, complex_activations_from_args
from prost_t2_classification.download import (
    DownloadEntry,
    presigned_url_expiration,
    select_entries_for_rawfiles,
    validate_download_url,
)
from prost_t2_classification.labels import (
    parse_split_exam_counts,
    select_middle_slices,
    select_preprocessing_labels,
    select_split_exams,
)


def _labels() -> pd.DataFrame:
    rows = []
    patient_id = 1
    for split, exams in (("training", 3), ("validation", 2), ("test", 2)):
        for exam_index in range(exams):
            rawfile = f"file_prostate_AXT2_{split}_{exam_index:03d}.h5"
            for slice_index in (1, 2):
                rows.append(
                    {
                        "fastmri_pt_id": patient_id,
                        "slice": slice_index,
                        "PIRADS": 3,
                        "label": 1,
                        "fastmri_rawfile": rawfile,
                        "data_split": split,
                        "folder": split,
                    }
                )
            patient_id += 1
    return pd.DataFrame(rows)


def test_select_split_exams_keeps_all_slices_for_requested_exam_counts():
    selected = select_split_exams(_labels(), {"training": 2, "validation": 1, "test": 1})

    exam_counts = selected.groupby("data_split")["fastmri_rawfile"].nunique().to_dict()
    assert exam_counts == {"test": 1, "training": 2, "validation": 1}
    assert selected.groupby(["data_split", "fastmri_rawfile"])["slice"].nunique().eq(2).all()


def test_select_split_exams_accepts_val_alias_and_raises_when_short():
    assert parse_split_exam_counts("train=1,val=1,test=1") == {
        "training": 1,
        "validation": 1,
        "test": 1,
    }
    with pytest.raises(ValueError, match="Requested 4 training exams"):
        select_split_exams(_labels(), {"training": 4})


def test_select_entries_for_rawfiles_matches_outputs_and_urls():
    entries = [
        DownloadEntry("https://example.test/a/file_prostate_AXT2_001.h5", "downloads/one.h5", "t2"),
        DownloadEntry("https://example.test/a/other.h5", "file_prostate_AXT2_002.h5", "t2"),
        DownloadEntry("https://example.test/a/prostate_t2.tar.gz", "prostate_t2.tar.gz", "t2"),
    ]

    selected = select_entries_for_rawfiles(
        entries,
        {"file_prostate_AXT2_001.h5", "file_prostate_AXT2_002.h5"},
    )

    assert selected == entries[:2]


def test_presigned_url_expiration_supports_s3_expiry_formats():
    assert presigned_url_expiration("https://example.test/file.h5?Expires=1782750295") == datetime(
        2026,
        6,
        29,
        16,
        24,
        55,
        tzinfo=timezone.utc,
    )
    assert presigned_url_expiration(
        "https://example.test/file.h5?X-Amz-Date=20260705T100000Z&X-Amz-Expires=60"
    ) == datetime(2026, 7, 5, 10, 1, tzinfo=timezone.utc)


def test_validate_download_url_raises_for_expired_signed_url():
    entry = DownloadEntry(
        "https://example.test/labels.tar.gz?Expires=1",
        "labels.tar.gz",
        "labels",
    )

    with pytest.raises(ValueError, match="expired on 1970-01-01 00:00:01 UTC"):
        validate_download_url(entry)


def test_run_accepts_separate_labels_path():
    args = build_parser().parse_args(
        [
            "run",
            "--light",
            "--skip-download",
            "--extract-dir",
            "D:/fastmri_prostate/T2",
            "--labels",
            "D:/fastmri_prostate/labels",
            "--recon-dir",
            "D:/fastmri_prostate_light/recon_t2",
            "--npz-dir",
            "D:/fastmri_prostate_light/npz_t2_coils",
            "--runs-dir",
            "D:/fastmri_prostate_light/runs",
        ]
    )

    assert args.labels == Path("D:/fastmri_prostate/labels")


def test_select_middle_slices_keeps_one_middle_label_per_exam():
    selected = select_middle_slices(_labels())

    assert selected.groupby(["folder", "fastmri_rawfile"]).size().eq(1).all()
    assert selected["slice"].eq(2).all()


def test_select_preprocessing_labels_keeps_all_slices_without_light():
    selected = select_preprocessing_labels(_labels())

    assert selected.groupby(["folder", "fastmri_rawfile"])["slice"].nunique().eq(2).all()
    assert len(selected) == len(_labels())


def test_select_preprocessing_labels_applies_split_counts_then_middle_slices_for_light():
    selected = select_preprocessing_labels(
        _labels(),
        split_exam_counts={"training": 2, "validation": 1, "test": 1},
        middle_slice_only=True,
    )

    exam_counts = selected.groupby("data_split")["fastmri_rawfile"].nunique().to_dict()
    assert exam_counts == {"test": 1, "training": 2, "validation": 1}
    assert len(selected) == 4
    assert selected["slice"].eq(2).all()


def test_reconstruct_accepts_labels_and_light_counts():
    args = build_parser().parse_args(
        [
            "reconstruct",
            "--raw-root",
            "D:/fastmri_prostate/T2",
            "--labels",
            "D:/fastmri_prostate/labels",
            "--recon-dir",
            "D:/fastmri_prostate_light/recon_t2",
            "--light",
            "--light-counts",
            "training=1,validation=1,test=1",
        ]
    )

    assert args.labels == Path("D:/fastmri_prostate/labels")
    assert args.light is True


def test_light_training_defaults_to_all_complex_activations():
    args = build_parser().parse_args(
        [
            "run",
            "--light",
            "--skip-download",
            "--skip-reconstruct",
            "--skip-npz",
            "--extract-dir",
            "D:/fastmri_prostate/T2",
            "--npz-dir",
            "D:/fastmri_prostate_light/npz_t2_coils",
            "--runs-dir",
            "D:/fastmri_prostate_light/runs",
        ]
    )

    assert complex_activations_from_args(args, light_mode=True) == ("modrelu", "crelu", "cardioid")


def test_regular_training_defaults_to_one_complex_activation_and_accepts_all():
    parser = build_parser()
    regular = parser.parse_args(
        [
            "train",
            "--manifest",
            "D:/fastmri_prostate/npz_t2_coils/manifest.csv",
            "--runs-dir",
            "D:/fastmri_prostate/runs",
        ]
    )
    all_activations = parser.parse_args(
        [
            "train",
            "--manifest",
            "D:/fastmri_prostate/npz_t2_coils/manifest.csv",
            "--runs-dir",
            "D:/fastmri_prostate/runs",
            "--complex-activations",
            "all",
        ]
    )

    assert complex_activations_from_args(regular) == ("modrelu",)
    assert complex_activations_from_args(all_activations) == ("modrelu", "crelu", "cardioid")


def test_reconstruct_selected_t2_file_writes_single_acquisition_slice(tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("fastmri_tools")
    from fastmri_tools.prostate_opts.fft import centered_ifft
    from fastmri_tools.prostate_opts.grappa import grappa_fill

    from prost_t2_classification.preprocess import reconstruct_selected_t2_file

    input_path = tmp_path / "file_prostate_AXT2_001.h5"
    output_path = tmp_path / "file_prostate_AXT2_001_complex_recon.h5"
    kspace = (
        np.ones((3, 4, 2, 4, 4), dtype=np.float32)
        + 1j * np.ones((3, 4, 2, 4, 4), dtype=np.float32)
    ).astype(np.complex64)
    calibration = np.ones((4, 2, 4, 4), dtype=np.complex64)

    with h5py.File(input_path, "w") as h5:
        h5.create_dataset("kspace", data=kspace)
        h5.create_dataset("calibration_data", data=calibration)

    result = reconstruct_selected_t2_file(
        input_path,
        output_path,
        centered_ifft=centered_ifft,
        grappa_fill=grappa_fill,
        kernel_size=(3, 3),
        slice_numbers_one_based=(3,),
    )

    assert result.image_complex_shape == (1, 1, 2, 4, 4)
    with h5py.File(output_path, "r") as h5:
        assert h5["image_complex"].shape == (1, 1, 2, 4, 4)
        assert "kspace_regridded" not in h5
        assert "kspace_grappa" not in h5
        assert h5.attrs["subset_reconstruction"] == np.True_
        assert h5.attrs["selected_acquisition_index"] == 1
        assert h5.attrs["selected_slice_indices"].tolist() == [2]


def test_reconstruct_t2_dataset_skips_failed_files(tmp_path, monkeypatch):
    fft_module = ModuleType("fastmri_tools.prostate_opts.fft")
    grappa_module = ModuleType("fastmri_tools.prostate_opts.grappa")
    fft_module.centered_ifft = lambda kspace, axes=(-2, -1): kspace
    grappa_module.grappa_fill = lambda kspace, calibration, kernel_size=(5, 5): kspace
    tqdm_module = ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, desc=None: iterable
    monkeypatch.setitem(sys.modules, "h5py", ModuleType("h5py"))
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_module)
    monkeypatch.setitem(sys.modules, "fastmri_tools", ModuleType("fastmri_tools"))
    monkeypatch.setitem(sys.modules, "fastmri_tools.prostate_opts", ModuleType("fastmri_tools.prostate_opts"))
    monkeypatch.setitem(sys.modules, "fastmri_tools.prostate_opts.fft", fft_module)
    monkeypatch.setitem(sys.modules, "fastmri_tools.prostate_opts.grappa", grappa_module)
    from prost_t2_classification import preprocess
    from prost_t2_classification.preprocess import ReconstructionResult

    raw_root = tmp_path / "raw"
    recon_root = tmp_path / "recon"
    folder = raw_root / "fastMRI_prostate_T2_IDS_001_020"
    folder.mkdir(parents=True)
    bad_input = folder / "file_prostate_AXT2_001.h5"
    good_input = folder / "file_prostate_AXT2_002.h5"
    bad_input.touch()
    good_input.touch()

    def fake_reconstruct(input_path, output_path, **kwargs):
        if input_path == bad_input:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"partial")
            raise RuntimeError("corrupt hdf5")
        output_path.write_bytes(b"ok")
        return ReconstructionResult(
            source_path=input_path,
            output_path=output_path,
            original_shape=(3, 4, 2, 4, 4),
            image_complex_shape=(1, 1, 2, 4, 4),
            acquisition_index=1,
            slice_indices=(2,),
        )

    monkeypatch.setattr(preprocess, "reconstruct_selected_t2_file", fake_reconstruct)

    outputs = preprocess.reconstruct_t2_dataset(raw_root, recon_root)

    bad_output = recon_root / "fastMRI_prostate_T2_IDS_001_020" / "file_prostate_AXT2_001_complex_recon.h5"
    good_output = recon_root / "fastMRI_prostate_T2_IDS_001_020" / "file_prostate_AXT2_002_complex_recon.h5"
    assert outputs == [good_output]
    assert not bad_output.exists()
    assert good_output.exists()
    failures = pd.read_csv(recon_root / "failed_reconstructions.csv")
    assert failures["input_path"].tolist() == [str(bad_input)]
    assert failures["error_type"].tolist() == ["RuntimeError"]


def test_make_npz_dataset_skips_missing_reconstructions(tmp_path):
    h5py = pytest.importorskip("h5py")
    from prost_t2_classification.preprocess import make_npz_dataset

    labels_path = tmp_path / "t2_slice_level_labels.csv"
    folder = "training"
    rows = []
    for patient_id, rawfile in (
        (1, "file_prostate_AXT2_001.h5"),
        (2, "file_prostate_AXT2_002.h5"),
    ):
        for slice_one_based in (1, 2):
            rows.append(
                {
                    "fastmri_pt_id": patient_id,
                    "slice": slice_one_based,
                    "PIRADS": 3,
                    "fastmri_rawfile": rawfile,
                    "data_split": "training",
                    "folder": folder,
                }
            )
    pd.DataFrame(rows).to_csv(labels_path, index=False)

    recon_root = tmp_path / "recon"
    recon_path = recon_root / folder / "file_prostate_AXT2_001_complex_recon.h5"
    recon_path.parent.mkdir(parents=True)
    image_complex = np.zeros((1, 2, 3, 4, 4), dtype=np.complex64)
    image_complex[:, :, 0] = 1 + 0j
    image_complex[:, :, 1] = 7 + 3j
    image_complex[:, :, 2] = 12 + 0j
    with h5py.File(recon_path, "w") as h5:
        h5.create_dataset("image_complex", data=image_complex)
        h5.attrs["selected_acquisition_index"] = 1
        h5.attrs["selected_slice_indices"] = np.array([0, 1], dtype=np.int32)

    npz_root = tmp_path / "npz"
    manifest_path = make_npz_dataset(labels_path, recon_root, npz_root, crop_size=4, max_coils=1)

    manifest = pd.read_csv(manifest_path)
    assert manifest["fastmri_rawfile"].tolist() == [
        "file_prostate_AXT2_001.h5",
        "file_prostate_AXT2_001.h5",
    ]
    assert manifest["slice"].tolist() == [1, 2]
    assert manifest["selected_coils"].astype(str).tolist() == ["1", "1"]
    with np.load(npz_root / manifest.iloc[0]["path"]) as npz:
        assert npz["image_complex"].shape == (1, 4, 4)
        assert np.all(npz["image_complex"] == np.complex64(7 + 3j))
    failures = pd.read_csv(npz_root / "failed_npz.csv")
    assert failures["fastmri_rawfile"].tolist() == ["file_prostate_AXT2_002.h5"]
    assert failures["error_type"].tolist() == ["FileNotFoundError"]
