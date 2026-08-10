from __future__ import annotations

import pandas as pd


def patient_split_sets(labels: pd.DataFrame) -> dict[str, set[int]]:
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
                raise ValueError(
                    f"Patient leakage between {left} and {right}: {sorted(overlap)[:10]}"
                )
