from pathlib import Path


def test_cluster_workflow_runs_real_and_all_complex_pooling_variants():
    script = (
        Path(__file__).parents[1] / "scripts" / "run_cluster_experiment.sh"
    ).read_text(encoding="utf-8")

    assert 'EXPERIMENT_NAME="${EXPERIMENT_NAME:-pooling_vs_real_v1}"' in script
    assert 'COMPLEX_POOLINGS="max median average"' in script
    assert 'EPOCHS="${EPOCHS:-100}"' in script
    assert 'TIME_LIMIT="${TIME_LIMIT:-48:00:00}"' in script
    assert 'MODEL_COUNT=4' in script
    assert 'array_last=$((PHASE2_SEEDS * MODEL_COUNT - 1))' in script
    assert 'seed_index=$((task_index / MODEL_COUNT))' in script
    assert 'model_index=$((task_index % MODEL_COUNT))' in script
    assert '"${train_args[@]}" --mode real' in script
    assert '"${train_args[@]}" --mode complex --complex-pooling "${pooling}"' in script
    assert 'completion_marker="${model_key^^}_COMPLETE"' in script
    assert '*_complex_modrelu/test_metrics.json' in script
    assert '*_complex_modrelu_median_pool/test_metrics.json' in script
    assert '*_complex_modrelu_average_pool/test_metrics.json' in script
