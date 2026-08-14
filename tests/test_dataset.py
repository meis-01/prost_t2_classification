import numpy as np
import pandas as pd
import pytest
import torch

from prost_t2_classification.dataset import (
    T2CoilNPZDataset,
    make_dataloaders,
    validate_manifest,
)
from prost_t2_classification.image_ops import (
    align_multicoil_phase,
    centered_fft2,
    scale_complex_by_magnitude,
)


def _valid_manifest() -> pd.DataFrame:
    rows = []
    patient = 1
    for split in ("training", "validation", "test"):
        for label in (0, 1):
            rows.append(
                {
                    "path": f"samples/{split}_{label}.npz",
                    "fastmri_pt_id": patient,
                    "label": label,
                    "data_split": split,
                    "channels": 4,
                }
            )
            patient += 1
    return pd.DataFrame(rows)


def test_validate_manifest_accepts_experiment_contract():
    validate_manifest(_valid_manifest())


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("label", 0.5, "binary 0/1"),
        ("channels", 4.5, "exactly four channels"),
        ("fastmri_pt_id", 1.5, "finite integers"),
        ("data_split", "holdout", "exactly the training"),
        ("path", "../outside.npz", "relative and contained"),
    ],
)
def test_validate_manifest_rejects_invalid_contract(column, value, message):
    manifest = _valid_manifest()
    manifest[column] = manifest[column].astype(object)
    manifest.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_manifest(manifest)


def test_balanced_sampler_sequence_depends_only_on_experiment_seed(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    _valid_manifest().to_csv(manifest_path, index=False)

    torch.manual_seed(1)
    first = make_dataloaders(
        manifest_path,
        mode="real",
        batch_size=2,
        num_workers=0,
        seed=10383,
    )
    torch.manual_seed(999)
    second = make_dataloaders(
        manifest_path,
        mode="complex",
        batch_size=2,
        num_workers=0,
        seed=10383,
    )

    assert list(first.train.sampler) == list(second.train.sampler)


def test_kspace_mode_transforms_prepared_complex_images(tmp_path):
    manifest = _valid_manifest()
    samples = tmp_path / "samples"
    samples.mkdir()
    rng = np.random.default_rng(73191)
    images = {}
    for row in manifest.itertuples():
        image = (
            rng.standard_normal((4, 8, 8))
            + 1j * rng.standard_normal((4, 8, 8))
        ).astype(np.complex64)
        images[row.path] = image
        np.savez_compressed(tmp_path / row.path, image_complex=image)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    dataset = T2CoilNPZDataset(
        manifest_path,
        split="training",
        mode="complex_kspace",
    )
    tensor, _ = dataset[0]

    expected = centered_fft2(align_multicoil_phase(images[dataset.rows.iloc[0]["path"]]))
    expected = scale_complex_by_magnitude(expected, shared_scale=True)
    assert tensor.dtype == torch.complex64
    assert torch.allclose(tensor, torch.from_numpy(expected))
