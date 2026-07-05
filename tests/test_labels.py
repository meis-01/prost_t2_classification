import pandas as pd
import pytest

from prost_t2_classification.labels import assert_patient_split_disjoint


def test_patient_split_leakage_raises():
    labels = pd.DataFrame(
        {
            "fastmri_pt_id": [1, 1],
            "data_split": ["training", "test"],
        }
    )
    with pytest.raises(ValueError):
        assert_patient_split_disjoint(labels)
