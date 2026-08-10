import pandas as pd
import pytest
import torch

from prost_t2_classification.dataset import make_dataloaders, validate_manifest


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
