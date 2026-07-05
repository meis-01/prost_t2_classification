from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import pandas as pd


REQUIRED_T2_COLUMNS = {
    "fastmri_pt_id",
    "slice",
    "PIRADS",
    "fastmri_rawfile",
    "data_split",
    "folder",
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
