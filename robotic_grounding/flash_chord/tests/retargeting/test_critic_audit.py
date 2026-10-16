"""Protect paired-budget joins and retrospective-vs-selected interpretations."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('critic_audit',
    Path(__file__).resolve().parents[2]/'scripts/audit_upper_critic.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def record():
    checkpoints=[]
    for budget, cost in [(100,1.), (200,3.)]:
        checkpoints.append(dict(environment_steps=budget, readiness_reasons=[],
            candidate=dict(cost=cost, training=dict(environment_steps=budget)),
            control=dict(cost=2., training=dict(environment_steps=budget))))
    return dict(step=1, upper_round=1, candidates=[dict(window=dict(window_id=0),
        predicted_wrench_return_gain=.01, actual_local_cost_gain=-1., accepted=False,
        rejection_reasons=['adaptation_comparison_not_stable_within_budget'],
        adaptation=dict(checkpoints=checkpoints, selected_environment_steps=200))])


def test_audit_distinguishes_selected_outcomes_from_best_earlier_checkpoint():
    rows=audit.proposal_rows([record()],43)
    result=audit.summarize(rows)
    assert result['predicted_improvements'] == 1
    assert result['measured_selected_improvements'] == 0
    assert result['retrospective_best_checkpoint_improvements'] == 1
    assert result['positive_prediction_precision'] == 0
    assert result['accepted'] == 0


def test_audit_rejects_mismatched_candidate_control_budgets():
    row=record()
    row['candidates'][0]['adaptation']['checkpoints'][0]['control']['training']['environment_steps']=99
    with pytest.raises(ValueError, match='budgets differ'):
        audit.proposal_rows([row],43)
