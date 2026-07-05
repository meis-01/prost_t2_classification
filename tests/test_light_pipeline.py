import pandas as pd
import pytest

from prost_t2_classification.download import DownloadEntry, select_entries_for_rawfiles
from prost_t2_classification.labels import (
    parse_split_exam_counts,
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
