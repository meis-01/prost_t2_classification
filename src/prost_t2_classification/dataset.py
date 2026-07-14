from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .image_ops import scale_complex_by_magnitude, standardize_real
from .labels import assert_patient_split_disjoint


Mode = Literal["real", "complex", "complex_kspace"]


class T2CoilNPZDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        mode: Mode,
        normalize: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.mode = mode
        self.normalize = normalize

        manifest = pd.read_csv(self.manifest_path)
        assert_patient_split_disjoint(manifest)
        split_mask = manifest["data_split"].astype(str).str.lower() == split.lower()
        self.rows = manifest[split_mask].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows found for split {split!r} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        sample_path = self.root / row["path"]
        if self.mode == "real":
            with np.load(sample_path) as npz:
                image_complex = npz["image_complex"].astype(np.complex64)
            image = np.abs(image_complex).astype(np.float32)
            if self.normalize:
                image = standardize_real(image)
            tensor = torch.from_numpy(image)
        elif self.mode == "complex":
            with np.load(sample_path) as npz:
                image_complex = npz["image_complex"].astype(np.complex64)
            image_complex = image_complex.astype(np.complex64)
            if self.normalize:
                image_complex = scale_complex_by_magnitude(image_complex)
            tensor = torch.from_numpy(image_complex)
        elif self.mode == "complex_kspace":
            with np.load(sample_path) as npz:
                kspace_complex = npz["kspace_complex"].astype(np.complex64)
            if self.normalize:
                kspace_complex = scale_complex_by_magnitude(kspace_complex)
            tensor = torch.from_numpy(kspace_complex)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return tensor, label


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    pos_weight: float


def make_dataloaders(
    manifest_path: Path,
    *,
    mode: Mode,
    batch_size: int,
    num_workers: int,
    normalize: bool = True,
) -> LoaderBundle:
    train_ds = T2CoilNPZDataset(manifest_path, split="training", mode=mode, normalize=normalize)
    val_ds = T2CoilNPZDataset(manifest_path, split="validation", mode=mode, normalize=normalize)
    test_ds = T2CoilNPZDataset(manifest_path, split="test", mode=mode, normalize=normalize)

    train_labels = train_ds.rows["label"].astype(int).to_numpy()
    positives = int(train_labels.sum())
    negatives = int(train_labels.shape[0] - positives)
    pos_weight = float(negatives / max(positives, 1))

    return LoaderBundle(
        train=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        validation=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        test=DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        pos_weight=pos_weight,
    )
