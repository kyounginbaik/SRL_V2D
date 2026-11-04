"""Freshly evaluate an existing checkpoint under the shared evaluator build."""
from pathlib import Path
import argparse
import json
import numpy as np
from evaluate_policy_component import evaluate


def run(baseline_report, output, episode_dir=None):
    baseline_report, output = Path(baseline_report), Path(output)
    original = json.loads(baseline_report.read_text())
    name = original['episode']
    directory = Path(episode_dir or original.get('input_dir') or
                     Path(__file__).resolve().parents[1]/'episodes'/name)
    result = evaluate(int(name.split('_')[-1]), original['execution']['scene'],
                      original['policy_checkpoint'], output, str(directory))
    comparison = {}
    with np.load(original['execution']['trajectory'], allow_pickle=False) as before, \
         np.load(result['execution']['trajectory'], allow_pickle=False) as after:
        for key in ['qpos', 'qvel', 'ctrl', 'time_s', 'substep_qpos', 'substep_qvel',
                    'substep_ctrl', 'substep_time_s', 'initial_state', 'final_state']:
            comparison[key] = dict(exact=bool(np.array_equal(before[key], after[key])),
                maximum_difference=float(np.max(np.abs(before[key]-after[key]), initial=0)))
    report = dict(baseline_report=str(baseline_report.resolve()),
                  actor=result['policy_architecture'], fields=comparison,
                  all_physical_states_actions_exact=all(x['exact'] for x in comparison.values()))
    (output/'shared_evaluator_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    if not report['all_physical_states_actions_exact']:
        raise RuntimeError('Shared evaluator changed physical execution; original component preserved')
    print(json.dumps(dict(output=str(output), exact=True,
                          proxy_mp_sr=result['metrics']['reference_proxy_mp_sr'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-report', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--episode-dir')
    run(**vars(parser.parse_args()))
