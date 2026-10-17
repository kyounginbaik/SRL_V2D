"""Audit and render a learned policy using its explicit reconstruction adapter."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
from audit_full_joint_limits import main as audit_limits
from grasp_stability import audit_grasp_stability


def run(run, baseline_report):
    run, baseline_report = Path(run), Path(baseline_report)
    result = json.loads((run/'result.json').read_text())
    base = json.loads(baseline_report.read_text())
    ex, baseline = result['execution'], base.get('execution', base)
    if not (result['learned_policy'] and ex['independent_control_replay']['ok']):
        raise ValueError('Expected verified online learned-policy execution')
    audit_limits(run)
    out = run/'audit'
    grasp = audit_grasp_stability(ex['scene'], ex['trajectory'], frames_only=False)
    (out/'grasp_stability_substeps.json').write_text(json.dumps(grasp, indent=2)+'\n')
    videos = run/'videos'
    videos.mkdir(exist_ok=True)
    args = dict(episode_dir=result['input_dir'],
        reference_scene=ex['scene'],
        baseline_report=str(Path(baseline['trajectory']).parent/'execution_report.json'),
        refined_report=str(Path(ex['trajectory']).parent/'execution_report.json'),
        out_dir=str(videos), width=512, height=384, stride=1, include_source=True)
    (videos/'render_args.json').write_text(json.dumps(args, indent=2)+'\n')
    subprocess.run([sys.executable, str(Path(__file__).with_name('render_rollouts.py')),
                    '--args', str(videos/'render_args.json')], check=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--baseline-report', required=True)
    run(**vars(p.parse_args()))
