"""Audit historical proposal usefulness without calling mixed task scores Q labels.

No simulator or training dependency is required. Saved proposal predictions are
wrench-return surrogates on fixed baseline states after a short actor response;
physical scores are mixed task costs after longer full-SAC adaptation. Their
association measures proposal usefulness, not same-objective critic accuracy or
gradient correctness. Best-checkpoint results are retrospective selection only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def proposal_rows(rounds, seed):
    rows = []
    for round_ in rounds:
        for candidate in round_['candidates']:
            if 'predicted_wrench_return_gain' not in candidate:
                continue
            checkpoints = candidate['adaptation']['checkpoints']
            gains = [p['control']['cost']-p['candidate']['cost'] for p in checkpoints]
            selected_budget = candidate['adaptation']['selected_environment_steps']
            selected = [p for p in checkpoints if p['environment_steps'] == selected_budget]
            if len(selected) != 1:
                raise ValueError('Selected checkpoint must identify exactly one paired budget')
            physical_gain = selected[0]['control']['cost']-selected[0]['candidate']['cost']
            if not np.isclose(physical_gain, candidate['actual_local_cost_gain']):
                raise ValueError('Logged gain does not match the selected paired checkpoint')
            for checkpoint in checkpoints:
                for branch in ('candidate', 'control'):
                    if checkpoint[branch]['training']['environment_steps'] != checkpoint['environment_steps']:
                        raise ValueError('Candidate/control training budgets differ')
            rows.append(dict(seed=seed, step=round_['step'], round=round_['upper_round'],
                window=candidate['window']['window_id'],
                predicted_support_gain=candidate['predicted_wrench_return_gain'],
                measured_task_gain_selected=physical_gain,
                measured_task_gain_first=gains[0], measured_task_gain_final=gains[-1],
                measured_task_gain_best=max(gains),
                selected_budget=selected_budget, selected_ready=not selected[0]['readiness_reasons'],
                accepted=bool(candidate['accepted']),
                plateau_rejected='adaptation_comparison_not_stable_within_budget' in candidate['rejection_reasons']))
    return rows


def summarize(rows):
    x = np.array([r['predicted_support_gain'] for r in rows])
    y = np.array([r['measured_task_gain_selected'] for r in rows])
    positive = x > 0
    rho = float(spearmanr(x, y).statistic) if len(set(x)) > 1 and len(set(y)) > 1 else None
    return dict(proposals=len(rows), predicted_improvements=int(positive.sum()),
        measured_selected_improvements=int((y > 0).sum()),
        predicted_and_measured_improvements=int(((y > 0) & positive).sum()),
        positive_prediction_precision=float(np.mean(y[positive] > 0)) if positive.any() else None,
        unconditional_measured_improvement_rate=float(np.mean(y > 0)),
        descriptive_spearman_different_objectives=rho,
        median_predicted_support_gain=float(np.median(x)),
        median_measured_task_gain=float(np.median(y)),
        first_checkpoint_improvements=sum(r['measured_task_gain_first'] > 0 for r in rows),
        final_checkpoint_improvements=sum(r['measured_task_gain_final'] > 0 for r in rows),
        retrospective_best_checkpoint_improvements=sum(r['measured_task_gain_best'] > 0 for r in rows),
        accepted=sum(r['accepted'] for r in rows),
        plateau_rejected=sum(r['plateau_rejected'] for r in rows))


def calibration(rounds):
    feedback = [f for r in rounds for f in r['critic_feedback'] if not f.get('trial_critic_skipped', 1)]
    return dict(rounds=len(rounds),
        median_row_holdout_return_mae_before=float(np.median([
            f['trial_critic_heldout_return_mae_before'] for f in feedback])),
        median_row_holdout_return_mae_after=float(np.median([
            f['trial_critic_heldout_return_mae_after'] for f in feedback])),
        median_td_mae=float(np.median([r['critic']['critic_heldout_td_mae'] for r in rounds])),
        median_feedback_fits_per_round=float(np.median([len(r['critic_feedback']) for r in rounds])),
        median_feedback_optimizer_updates_per_round=float(np.median([
            sum(f.get('trial_critic_updates', 0) for f in r['critic_feedback']) for r in rounds])),
        median_feedback_fit_seconds=float(np.median([
            sum(f['critic_fit_seconds'] for f in r['critic_feedback']) for r in rounds])),
        median_round_seconds=float(np.median([r['seconds'] for r in rounds])))


def plot(rows, summaries, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout='constrained')
    colors = ['#2874A6', '#D68910', '#148F77']
    for color, (seed, summary) in zip(colors, summaries.items()):
        selected = [r for r in rows if r['seed'] == int(seed)]
        axes[0].scatter([r['predicted_support_gain'] for r in selected],
            [r['measured_task_gain_selected'] for r in selected],
            alpha=.5, s=14, color=color, label=f'Seed {seed}')
    axes[0].axhline(0, color='black', lw=.8); axes[0].axvline(0, color='black', lw=.8)
    axes[0].set(xlabel='Predicted support-return gain', ylabel='Measured mixed task-score gain',
        title='Proposal usefulness (different objectives)')
    axes[0].legend(fontsize=8)
    positions = np.arange(len(summaries))
    for offset, key, label, color in [(-.25, 'first_checkpoint_improvements', 'First budget', '#AAB7B8'),
            (0, 'final_checkpoint_improvements', 'Final budget', '#2874A6'),
            (.25, 'retrospective_best_checkpoint_improvements', 'Best of tested budgets', '#D68910')]:
        axes[1].bar(positions+offset, [s[key]/s['proposals'] for s in summaries.values()],
            width=.24, label=label, color=color)
    axes[1].set(xticks=positions, xticklabels=list(summaries), ylim=(0,1),
        xlabel='Seed', ylabel='Fraction beating matched control', title='Checkpoint choice changes outcomes')
    axes[1].legend(fontsize=8)
    for offset, key, label, color in [(-.18, 'median_predicted_support_gain', 'Predicted gain', '#2874A6'),
            (.18, 'median_row_holdout_return_mae_after', 'Return MAE after fitting', '#D68910')]:
        axes[2].bar(positions+offset, [s[key] for s in summaries.values()], width=.35, color=color, label=label)
    axes[2].set(xticks=positions, xticklabels=list(summaries), xlabel='Seed',
        ylabel='Normalized critic units', title='Signal size vs. row-calibration error')
    axes[2].legend(fontsize=8)
    fig.suptitle('Historical Adam-feedback runs: 405 proposals, 132 outer rounds')
    fig.savefig(output/'critic_audit.png', dpi=180)
    fig.savefig(output/'critic_audit.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Historical experiment root')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows, summaries, provenance = [], {}, []
    for seed in (43, 44, 45):
        path = args.root/f'episode_000001/runs/pilot/episode_000001/chunk/seed_{seed}/training/upper_updates.jsonl'
        data = path.read_bytes()
        if not data.endswith(b'\n'):
            raise ValueError(f'Incomplete JSONL: {path}')
        rounds = [json.loads(line) for line in data.splitlines() if line]
        rows = proposal_rows(rounds, seed)
        summaries[str(seed)] = {**summarize(rows), **calibration(rounds)}
        all_rows.extend(rows)
        provenance.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest()))
    report = dict(schema_version=1, aggregate=summarize(all_rows), per_seed=summaries,
        inputs=provenance, limitations=[
            'Predictions and physical task scores use different objectives, states and lower-response lengths.',
            'No saved per-frame costs or pre-proposal critic snapshots: exact return calibration and physical gradient agreement cannot be recovered.',
            'Within-rollout row holdouts are calibration diagnostics, not independent trajectories.',
            'Trials share rounds and controllers; counts and correlations are descriptive, not independent-sample significance tests.',
            'Best-checkpoint gains are retrospective and do not certify guards, confirmation or task success.'])
    (args.output/'critic_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with (args.output/'proposals.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    if args.plot:
        plot(all_rows, summaries, args.output)
    print(json.dumps(report['aggregate'], indent=2))


if __name__ == '__main__':
    main()
