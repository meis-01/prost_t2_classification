from pathlib import Path


def test_cluster_workflow_is_locked_to_real_vs_complex_median():
    script = (
        Path(__file__).parents[1] / "scripts" / "run_cluster_experiment.sh"
    ).read_text(encoding="utf-8")

    assert 'EXPERIMENT_NAME="${EXPERIMENT_NAME:-median_vs_real_v1}"' in script
    assert 'COMPLEX_POOLING="median"' in script
    assert '--mode both' in script
    assert '--complex-pooling "${COMPLEX_POOLING}"' in script
    assert '*_complex_modrelu_median_pool/test_metrics.json' in script
    assert "maxpool_v1" not in script
