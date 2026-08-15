from pathlib import Path


def test_cluster_workflow_launches_one_complete_array_per_seed():
    script = (
        Path(__file__).parents[1] / "scripts" / "run_cluster_experiment.sh"
    ).read_text(encoding="utf-8")

    assert 'EXPERIMENT_NAME="${EXPERIMENT_NAME:-complex_full_factorial_v1}"' in script
    assert "MODEL_COUNT=481" in script
    assert "COMPLEX_MODEL_COUNT=480" in script
    assert 'PHASE2_SEEDS="${PHASE2_SEEDS:-4}"' in script
    assert 'MAX_PARALLEL="${MAX_PARALLEL:-0}"' in script
    assert 'TIME_LIMIT="${TIME_LIMIT:-48:00:00}"' in script
    assert "model_array_spec()" in script
    assert "if (( MAX_PARALLEL == 0 ))" in script
    assert "printf '0-%s\\n'" in script
    assert "printf '0-%s%%%s\\n'" in script
    assert 'pilot_spec="$(model_array_spec)"' in script
    assert 'array_spec="$(model_array_spec)"' in script
    assert 'for ((seed_index = 0; seed_index < PHASE2_SEEDS; seed_index++))' in script
    assert '--export="ALL,PHASE2_SEED_INDEX=${seed_index}"' in script
    assert 'seed_index="${PHASE2_SEED_INDEX:?phase2 requires PHASE2_SEED_INDEX}"' in script
    assert 'model_index="${local_task}"' in script
    assert 'previous_job="${array_job}"' in script
    assert 'dependency_args=(--dependency="afterany:${previous_job}")' in script
    assert '--dependency="afterany:${previous_job}"' in script
    assert "afterok" not in script
    assert "pilot_to_phase2_dependency=afterany" in script
    assert "experiment_grid row --index" in script
    assert '--complex-input-domain "${input_domain}"' in script
    assert '--complex-normalization "${normalization}"' in script
    assert '--complex-convolution "${convolution}"' in script
    assert '--complex-streams "${streams}"' in script
    assert '--complex-interaction "${interaction}"' in script
    assert '--complex-activation "${activation}"' in script
    assert 'if [[ -f "${model_dir}/COMPLETE" ]]' in script
    assert 'touch "${EXPERIMENT_ROOT}/PHASE2_INCOMPLETE"' in script
    assert "validate_completed_pilots" not in script
    assert 'status) show_status' in script
    assert 'pending_or_queued=' in script
    assert "resume-phase2)" in script
