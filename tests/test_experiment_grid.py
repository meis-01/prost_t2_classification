import json
from collections import Counter

import pandas as pd

from prost_t2_classification.experiment_grid import experiment_model_specs, write_grid


def test_full_factorial_grid_is_complete_unique_and_keeps_existing_models_first():
    specs = experiment_model_specs()

    assert len(specs) == 481
    assert [spec.model_key for spec in specs[:4]] == [
        "real",
        "complex_max",
        "complex_median",
        "complex_average",
    ]
    assert len({spec.model_key for spec in specs}) == len(specs)

    complex_specs = specs[1:]
    assert Counter(spec.input_domain for spec in complex_specs) == {
        "image": 240,
        "kspace": 240,
    }
    assert Counter(spec.pooling for spec in complex_specs) == {
        "max": 160,
        "median": 160,
        "average": 160,
    }
    assert Counter(spec.activation for spec in complex_specs) == {
        "modrelu": 120,
        "magnitude_silu": 120,
        "crelu": 120,
        "cardioid": 120,
    }
    assert not any(
        spec.streams == "complex_only" and spec.interaction == "modulus_gate"
        for spec in complex_specs
    )


def test_grid_manifest_writer_records_every_model(tmp_path):
    csv_path = tmp_path / "model_grid.csv"
    json_path = tmp_path / "model_grid.json"

    write_grid(csv_path, json_path)

    assert len(pd.read_csv(csv_path)) == 481
    records = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(records) == 481
    assert records[0]["model_key"] == "real"
    assert records[-1]["model_key"] == "cx_ksp_avg_cbn_wl_dual_holo_card"
