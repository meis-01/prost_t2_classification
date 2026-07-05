from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Set

import pandas as pd


REQUIRED_T2_COLUMNS = {
    "fastmri_pt_id",
    "slice",
    "PIRADS",
    "fastmri_rawfile",
    "data_split",
    "folder",
}

DEFAULT_LIGHT_SPLIT_EXAM_COUNTS = {
    "training": 10,
    "validation": 5,
    "test": 5,
}

_SPLIT_ALIASES = {
    "train": "training",
    "training": "training",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}


def find_t2_labels(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.rglob("t2_slice_level_labels.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find t2_slice_level_labels.csv under {path}.")
    return matches[0]


def load_t2_labels(path: Path) -> pd.DataFrame:
    label_file = find_t2_labels(path)
    df = pd.read_csv(label_file)
    unnamed = [column for column in df.columns if column.startswith("Unnamed") or column == ""]
    if unnamed:
        df = df.drop(columns=unnamed)

    missing = REQUIRED_T2_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{label_file} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["fastmri_pt_id"] = df["fastmri_pt_id"].astype(int)
    df["slice"] = df["slice"].astype(int)
    df["PIRADS"] = df["PIRADS"].astype(int)
    df["label"] = (df["PIRADS"] > 2).astype(int)
    df["data_split"] = df["data_split"].astype(str).str.lower()
    return df


def parse_split_exam_counts(value: str | None) -> Dict[str, int]:
    if value is None:
        return dict(DEFAULT_LIGHT_SPLIT_EXAM_COUNTS)

    counts: Dict[str, int] = {}
    for item in value.split(","):
        if "=" not in item:
            raise ValueError("split counts must look like 'training=10,validation=5,test=5'.")
        split, raw_count = item.split("=", 1)
        split = normalize_split_name(split.strip())
        try:
            count = int(raw_count.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid exam count for {split!r}: {raw_count!r}") from exc
        if count < 0:
            raise ValueError("split exam counts must be non-negative.")
        counts[split] = count

    if not counts:
        raise ValueError("At least one split exam count is required.")
    return counts


def normalize_split_name(value: str) -> str:
    key = value.strip().lower()
    if key not in _SPLIT_ALIASES:
        raise ValueError(f"Unknown split {value!r}; expected training, validation, or test.")
    return _SPLIT_ALIASES[key]


def select_split_exams(labels: pd.DataFrame, split_counts: Mapping[str, int]) -> pd.DataFrame:
    normalized_counts = {
        normalize_split_name(split): int(count)
        for split, count in split_counts.items()
        if int(count) > 0
    }
    selected_keys = set()

    for split, count in normalized_counts.items():
        split_labels = labels[labels["data_split"].astype(str).str.lower() == split]
        exams = (
            split_labels[["folder", "fastmri_rawfile"]]
            .drop_duplicates()
            .sort_values(["folder", "fastmri_rawfile"])
            .head(count)
        )
        if len(exams) < count:
            raise ValueError(
                f"Requested {count} {split} exams, but only found {len(exams)} in the T2 labels."
            )
        selected_keys.update(
            normalize_exam_key(str(row.folder), str(row.fastmri_rawfile))
            for row in exams.itertuples(index=False)
        )

    selected = labels[
        labels.apply(
            lambda row: normalize_exam_key(str(row["folder"]), str(row["fastmri_rawfile"])) in selected_keys,
            axis=1,
        )
    ].copy()
    if selected.empty:
        raise ValueError("No labels remain after applying split exam counts.")
    return selected


def select_middle_slices(labels: pd.DataFrame) -> pd.DataFrame:
    indices = []
    for _, group in labels.groupby(["folder", "fastmri_rawfile"], sort=True):
        group = group.sort_values("slice")
        indices.append(int(group.index[len(group) // 2]))
    if not indices:
        return labels.copy()
    return labels.loc[indices].copy().reset_index(drop=True)


def select_preprocessing_labels(
    labels: pd.DataFrame,
    *,
    split_exam_counts: Mapping[str, int] | None = None,
    limit_patients: int | None = None,
    limit_slices: int | None = None,
) -> pd.DataFrame:
    assert_patient_split_disjoint(labels)

    if split_exam_counts is not None:
        labels = select_split_exams(labels, split_exam_counts)
    if limit_patients is not None:
        keep_patients = sorted(labels["fastmri_pt_id"].unique())[:limit_patients]
        labels = labels[labels["fastmri_pt_id"].isin(keep_patients)].copy()

    labels = select_middle_slices(labels)
    if limit_slices is not None:
        labels = labels.head(limit_slices).copy()
    return labels


def exam_keys_from_labels(labels: pd.DataFrame) -> Set[tuple[str, str]]:
    return {
        normalize_exam_key(str(row.folder), str(row.fastmri_rawfile))
        for row in labels[["folder", "fastmri_rawfile"]].drop_duplicates().itertuples(index=False)
    }


def rawfile_names_from_labels(labels: pd.DataFrame) -> Set[str]:
    return {rawfile for _, rawfile in exam_keys_from_labels(labels)}


def normalize_exam_key(folder: str, rawfile: str) -> tuple[str, str]:
    normalized_folder = Path(folder).as_posix().strip("/")
    if normalized_folder == ".":
        normalized_folder = ""
    return normalized_folder, Path(rawfile).name


def patient_split_sets(labels: pd.DataFrame) -> Dict[str, Set[int]]:
    return {
        split: set(group["fastmri_pt_id"].astype(int).tolist())
        for split, group in labels.groupby("data_split")
    }


def assert_patient_split_disjoint(labels: pd.DataFrame) -> None:
    split_sets = patient_split_sets(labels)
    splits = sorted(split_sets)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            overlap = split_sets[left].intersection(split_sets[right])
            if overlap:
                preview = sorted(overlap)[:10]
                raise ValueError(
                    f"Patient leakage between {left} and {right}: {preview}"
                )


def resolve_reconstruction_path(recon_root: Path, folder: str, rawfile: str) -> Path:
    expected = recon_root / folder / f"{Path(rawfile).stem}_complex_recon.h5"
    if expected.exists():
        return expected

    matches = sorted(recon_root.rglob(f"{Path(rawfile).stem}_complex_recon.h5"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find reconstruction for {folder}/{rawfile} under {recon_root}")
