from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
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
        augment_global_phase: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.mode = mode
        self.augment_global_phase = augment_global_phase

        manifest = pd.read_csv(self.manifest_path)
        validate_manifest(manifest)
        split_mask = manifest["data_split"].astype(str).str.strip().str.lower() == split.lower()
        self.rows = manifest[split_mask].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No rows found for split {split!r} in {manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        sample_path = self.root / str(row["path"])
        with np.load(sample_path) as npz:
            if "image_complex" not in npz:
                raise ValueError(f"{sample_path} does not contain image_complex.")
            raw_image = npz["image_complex"]
        if not np.iscomplexobj(raw_image):
            raise ValueError(f"{sample_path} image_complex must use a complex dtype.")
        image_complex = raw_image.astype(np.complex64)
        if image_complex.ndim != 3 or image_complex.shape[0] != 4:
            raise ValueError(f"{sample_path} image_complex must have shape (4, height, width).")
        if min(image_complex.shape[1:]) < 8:
            raise ValueError(f"{sample_path} spatial dimensions must both be at least 8 pixels.")
        if not np.isfinite(image_complex).all():
            raise ValueError(f"{sample_path} image_complex contains non-finite values.")

        image_complex = align_multicoil_phase(image_complex)
        image_complex = scale_complex_by_magnitude(image_complex, shared_scale=True)
        if self.mode == "complex" and self.augment_global_phase:
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


def validate_manifest(manifest: pd.DataFrame) -> None:
    required = {"path", "fastmri_pt_id", "label", "data_split", "channels"}
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if manifest.empty:
        raise ValueError("Manifest is empty.")

    try:
        labels = pd.to_numeric(manifest["label"], errors="raise")
        channels = pd.to_numeric(manifest["channels"], errors="raise")
        patient_ids = pd.to_numeric(manifest["fastmri_pt_id"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Manifest labels, channels, and patient IDs must be numeric.") from exc
    if labels.isna().any() or not labels.isin([0, 1]).all():
        found = sorted({str(value) for value in manifest["label"]})
        raise ValueError(f"Manifest labels must be binary 0/1; found {found}.")
    if channels.isna().any() or not channels.eq(4).all():
        found = sorted({str(value) for value in manifest["channels"]})
        raise ValueError(f"Manifest must contain exactly four channels per sample; found {found}.")
    if (
        patient_ids.isna().any()
        or not np.isfinite(patient_ids).all()
        or not patient_ids.eq(np.floor(patient_ids)).all()
    ):
        raise ValueError("Manifest patient IDs must be finite integers.")

    splits = manifest["data_split"].astype(str).str.strip().str.lower()
    required_splits = {"training", "validation", "test"}
    found_splits = set(splits)
    if found_splits != required_splits:
        raise ValueError(
            "Manifest must contain exactly the training, validation, and test splits; "
            f"found {sorted(found_splits)}."
        )
    for split in sorted(required_splits):
        split_labels = set(labels[splits == split].astype(int))
        if split_labels != {0, 1}:
            raise ValueError(f"Manifest {split} split must contain both binary classes.")

    if manifest["path"].isna().any() or manifest["path"].astype(str).str.strip().eq("").any():
        raise ValueError("Manifest sample paths must not be empty.")
    if manifest["path"].duplicated().any():
        raise ValueError("Manifest sample paths must be unique.")
    for value in manifest["path"]:
        text = str(value)
        path = Path(text)
        windows_path = PureWindowsPath(text)
        if (
            path.is_absolute()
            or windows_path.is_absolute()
            or ".." in path.parts
            or ".." in windows_path.parts
        ):
            raise ValueError(f"Manifest sample paths must be relative and contained in the data directory: {text}")
    assert_patient_split_disjoint(manifest)


def make_dataloaders(
    manifest_path: Path,
    *,
    mode: Mode,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> LoaderBundle:
    train_ds = T2CoilNPZDataset(
        manifest_path,
        split="training",
        mode=mode,
        augment_global_phase=mode == "complex",
    )
    val_ds = T2CoilNPZDataset(manifest_path, split="validation", mode=mode)
    test_ds = T2CoilNPZDataset(manifest_path, split="test", mode=mode)

    train_labels = train_ds.rows["label"].astype(int).to_numpy()
    positives = int(train_labels.sum())
    negatives = int(train_labels.shape[0] - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Training split must contain both positive and negative samples.")

    sample_weights = np.where(
        train_labels == 1,
        0.5 / positives,
        0.5 / negatives,
    )
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    common_loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }

    return LoaderBundle(
        train=DataLoader(
            train_ds,
            shuffle=False,
            sampler=sampler,
            generator=torch.Generator().manual_seed(seed + 1),
            **common_loader_options,
        ),
        validation=DataLoader(
            val_ds,
            shuffle=False,
            generator=torch.Generator().manual_seed(seed + 2),
            **common_loader_options,
        ),
        test=DataLoader(
            test_ds,
            shuffle=False,
            generator=torch.Generator().manual_seed(seed + 3),
            **common_loader_options,
        ),
    )
