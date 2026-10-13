"""Summarize matched frozen-policy probes and explain confirmation rejection."""
import argparse
import json
from pathlib import Path


def gate_failures(summary):
    validation = summary.get('confirmation')
    if not validation:
        return ['no eligible improving candidate selected']
    before, after = validation['baseline'], validation['candidate']
    failures = []
    for key, tolerance in [('object_position_robust_m', .002),
                           ('object_orientation_robust_rad', .01),
                           ('per_object_position_mean_m', .002),
                           ('per_object_orientation_mean_rad', .01),
                           ('per_object_contact_proximity_mean_m', .002)]:
        old, new = before[key], after[key]
        old = old if isinstance(old, list) else [old]
        new = new if isinstance(new, list) else [new]
        for index, (x, y) in enumerate(zip(old, new)):
            if (x is None) != (y is None) or (x is not None and y > x + tolerance):
                failures.append(f'{key}[{index}]')
    for h, (old, new) in enumerate(zip(before['per_hand_object_contact_proximity_mean_m'],
                                     after['per_hand_object_contact_proximity_mean_m'])):
        for o, (x, y) in enumerate(zip(old, new)):
            if (x is None) != (y is None) or (x is not None and y > x + .002):
                failures.append(f'contact_hand_{h}_object_{o}')
    if after['contact_proximity_mean_m'] > before['contact_proximity_mean_m'] - .001:
        failures.append('aggregate_contact_gain_below_1mm')
    if after['object_success_robust_fraction'] < before['object_success_robust_fraction'] - .01:
        failures.append('robust_success_regression')
    if after['unstable_world_fraction'] > before['unstable_world_fraction']:
        failures.append('stability_regression')
    if not after['all_errors_finite'] or not after['full_horizon']:
        failures.append('nonfinite_or_short_rollout')
    # Final acceptance also checks total cost, which is recorded in scores.json.
    if not summary['accepted'] and not failures:
        failures.append('confirmation_cost_did_not_improve')
    return failures


def summarize(root):
    rows, identities = [], set()
    for f in sorted(root.glob('*/seed_*/summary.json')):
        r = json.loads(f.read_text())
        config = json.loads(f.with_name('config.json').read_text())
        identities.add((config['checkpoint_sha256'], config['starting_reference_sha256'], config['candidate_count']))
        row = {k: r[k] for k in ('variant', 'seed', 'dimensions', 'accepted', 'gate_pass_count',
                                'improving_count', 'baseline_cost', 'best_raw_cost', 'best_valid_cost', 'seconds')}
        row['best_valid_cost_gain_percent'] = (100 * (1 - r['best_valid_cost'] / r['baseline_cost'])
                                              if r['best_valid_cost'] is not None else None)
        row['confirmation_failures'] = gate_failures(r)
        rows.append(row)
    if len(identities) > 1:
        raise ValueError('Results do not share checkpoint, starting reference, and candidate budget')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    payload = json.dumps(summarize(a.root), indent=2, allow_nan=False) + '\n'
    if a.output:
        a.output.write_text(payload)
    print(payload, end='')


if __name__ == '__main__':
    main()
