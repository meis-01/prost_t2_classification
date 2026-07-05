from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from prost_t2_classification.cli import build_parser
from prost_t2_classification.download import (
    DownloadEntry,
    presigned_url_expiration,
    select_entries_for_rawfiles,
    validate_download_url,
)
from prost_t2_classification.labels import (
    parse_split_exam_counts,
    select_middle_slices,
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
