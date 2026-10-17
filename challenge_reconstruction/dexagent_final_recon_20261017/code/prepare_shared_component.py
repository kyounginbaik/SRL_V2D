"""Stage one successful actor with a common evaluator manifest and portable replay.

This creates an episode component for review; it neither claims that the entire
roster is complete nor uploads anything. Visual/contact audits are preserved
separately from the reference-proxy pose metric.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import shutil
import subprocess
import sys
from stage_policy_component import prepare, stage


COMMON_CODE = [
    'component_policy.py', 'learned_feedback_policy.py',
    'posture_feedback_policy.py', 'posture_feedback_policy_v3.py',
    'posture_feedback_policy_v4.py', 'joint_margin_policy.py',
    'pinky_margin_policy_v2.py', 'floating_runtime.py', 'evaluate_mp_sr.py',
    'release_feedback_policy.py', 'wrist_reference.py',
    'grasp_stability.py', 'stage_policy_component.py', 'submission_readiness.py',
    'reproduce_policy_component.py', 'prepare_shared_component.py',
]


def run(args):
    code = Path(__file__).resolve().parent
    report = json.loads((args.run/'result.json').read_text())
    args.episode = int(report['episode'].split('_')[-1])
    args.episode_dir = args.episode_dir or Path(report['input_dir'])
    args.evaluator_script = code/'evaluate_policy_component.py'
    args.evaluation_code = [code/name for name in COMMON_CODE]
    args.output = args.output.resolve()
    prepared, manifest, roster = prepare(args.episode, args.output, args.episode_dir)
    result = stage(args, prepared, manifest, roster)
    if args.quality_dir:
        shutil.copytree(args.quality_dir, args.output/'validation/physical_quality', dirs_exist_ok=False)
    command = [sys.executable, str(code/'reproduce_policy_component.py'), 'build',
        '--bundle', str(args.output/'portable'), '--evaluation', str(args.run),
        '--checkpoint', str(args.checkpoint), '--training-report', str(args.training_report),
        '--episode-dir', str(args.episode_dir)]
    if args.base_training_report:
        command += ['--base-training-report', str(args.base_training_report)]
    for file in args.lineage_file:
        command += ['--lineage-file', str(file)]
    subprocess.run(command, check=True)
    print(json.dumps(dict(component=str(args.output), pose_rows=result['expected_trajectory_rows'],
        reference_proxy_mp_sr=result['reference_proxy_mp_sr'],
        portable_replay_pending=True, complete_submission_ready=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['run', 'checkpoint', 'training-report', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    for name in ['base-training-report', 'episode-dir', 'quality-dir']:
        p.add_argument('--'+name, type=Path)
    p.add_argument('--lineage-file', type=Path, action='append', default=[])
    run(p.parse_args())
