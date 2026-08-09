from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .image_ops import align_multicoil_phase, scale_complex_by_magnitude
from .labels import assert_patient_split_disjoint


Mode = Literal["real", "complex"]


class T2CoilNPZDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        mode: Mode,
        normalize: bool = True,
        random_global_phase: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.mode = mode
        self.normalize = normalize
        self.random_global_phase = random_global_phase

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
        with np.load(sample_path) as npz:
            image_complex = npz["image_complex"].astype(np.complex64)

        image_complex = align_multicoil_phase(image_complex)
        if self.normalize:
            image_complex = scale_complex_by_magnitude(image_complex, shared_scale=True)
        if self.mode == "complex" and self.random_global_phase:
            phase = np.random.uniform(-np.pi, np.pi)
            image_complex *= np.complex64(np.exp(1j * phase))

        if self.mode == "real":
            image = np.abs(image_complex).astype(np.float32)
            tensor = torch.from_numpy(image)
        elif self.mode == "complex":
            tensor = torch.from_numpy(image_complex)
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
    balanced_sampling: bool = True,
) -> LoaderBundle:
    train_ds = T2CoilNPZDataset(
        manifest_path,
        split="training",
        mode=mode,
        normalize=normalize,
        random_global_phase=mode == "complex",
    )
    val_ds = T2CoilNPZDataset(manifest_path, split="validation", mode=mode, normalize=normalize)
    test_ds = T2CoilNPZDataset(manifest_path, split="test", mode=mode, normalize=normalize)

    train_labels = train_ds.rows["label"].astype(int).to_numpy()
    positives = int(train_labels.sum())
    negatives = int(train_labels.shape[0] - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Training split must contain both positive and negative samples.")

    sampler = None
    shuffle = True
    pos_weight = float(negatives / positives)
    if balanced_sampling:
        sample_weights = np.where(
            train_labels == 1,
            0.5 / positives,
            0.5 / negatives,
        )
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle = False
        pos_weight = 1.0

    return LoaderBundle(
        train=DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
        ),
        validation=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        test=DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        pos_weight=pos_weight,
    )
