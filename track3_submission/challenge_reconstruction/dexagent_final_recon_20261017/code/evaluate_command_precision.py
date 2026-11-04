"""Evaluate explicitly versioned actuator-command precision, without scene edits."""
from pathlib import Path
import argparse
import json
import numpy as np
from learned_feedback_policy import sha
from evaluate_policy_component import evaluate
from audit_full_joint_limits import main as audit_joints


def run(episode, decimals, node=None, baseline_report=None, output=None):
    b = Path(__file__).resolve().parents[1]
    name = f'episode_{episode:06d}'
    base = b/'refinement_20261018/pinky_margin_v2'/name
    original = json.loads(Path(baseline_report or base/'evaluation/result.json').read_text())
    source_checkpoint = Path(original['policy_checkpoint'])
    if output is None and node is None:
        raise ValueError('Specify output or node')
    out = Path(output) if output else b/'refinement_20261018/command_precision'/name/f'decimals_{decimals}'/node
    out.mkdir(parents=True, exist_ok=True)
    if (out/'training_report.json').exists():
        raise FileExistsError(out)
    with np.load(source_checkpoint, allow_pickle=False) as z:
        weights = {k:z[k] for k in z.files}
    weights['component_output_decimals'] = np.array(decimals)
    checkpoint = out/'policy.npz'
    np.savez_compressed(checkpoint, **weights)
    report = dict(method='Versioned actuator command precision applied to a physically fine-tuned learned actor',
        source_checkpoint=str(source_checkpoint), source_checkpoint_sha256=sha(source_checkpoint),
        source_training_report=str(source_checkpoint.parent/'training_report.json'),
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        learned_weights_unchanged=True, command_decimal_places=decimals,
        source_targets_unchanged=True, initial_state_unchanged=True,
        physics_unchanged=True, reason='Test portability across CPU-dependent rotation roundoff')
    (out/'training_report.json').write_text(json.dumps(report, indent=2)+'\n')
    result = evaluate(episode, original['execution']['scene'], checkpoint, out/'evaluation', original['input_dir'])
    audit_joints(out/'evaluation')
    print(json.dumps(dict(episode=name, decimals=decimals, node=node,
        mp_sr=result['metrics']['reference_proxy_mp_sr'],
        objects=result['metrics']['declared_frame_reference_proxy']['objects'])))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--decimals', type=int, required=True)
    p.add_argument('--node')
    p.add_argument('--baseline-report', type=Path)
    p.add_argument('--output', type=Path)
    run(**vars(p.parse_args()))
