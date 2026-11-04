"""Distil any full-horizon physical teacher recording (trajectory.npz with substep ctrl/qpos/qvel) into the registered
radial-basis phase/proprioception actor and evaluate it with the UNCHANGED frozen evaluator in the episode's frozen scene.
  distill_teacher.py --episode E --teacher <dir with execution/trajectory.npz (+ execution/scene_physics.xml)> --output DIR
     [--feedback-gain G --centers C --decimals D --seed S]   (defaults: the episode's shipped actor hyperparameters)
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, hashlib
ROOT = Path(__file__).resolve().parents[1]; CODE = ROOT / 'code'; sys.path.insert(0, str(CODE))
from learned_feedback_policy import train, sha

ap = argparse.ArgumentParser(); ap.add_argument('--episode', type=int, required=True); ap.add_argument('--teacher', required=True); ap.add_argument('--output', required=True)
ap.add_argument('--centers', type=int); ap.add_argument('--feedback-gain', type=float); ap.add_argument('--velocity-gain', type=float); ap.add_argument('--decimals', type=int); ap.add_argument('--seed', type=int, default=43)
a = ap.parse_args(); e = a.episode
orig = ROOT / f'refinement_20261018/shared_policy_handoff/episode_{e:06d}/portable/original'
ev = json.loads((orig / 'evaluation_result.json').read_text()); scene = ev['execution']['scene']; episode_dir = ev['input_dir']
with np.load(orig / 'policy.npz', allow_pickle=False) as z:
    centers = a.centers or len(z['centers']); decimals = a.decimals if a.decimals is not None else (int(z['component_output_decimals']) if 'component_output_decimals' in z.files else None)
rep = {}
for f in ('base_training_report.json', 'training_report.json'):
    if (orig / f).exists():
        rep = json.loads((orig / f).read_text())
        if rep.get('feedback_teacher_position_gain') is not None: break
fg = a.feedback_gain or rep.get('feedback_teacher_position_gain', .03); vg = a.velocity_gain or rep.get('feedback_teacher_velocity_gain', .0005)
teacher = Path(a.teacher).resolve(); traj = teacher / 'execution/trajectory.npz'; tscene = teacher / 'execution/scene_physics.xml'
if sha(tscene) != sha(scene): raise SystemExit(f'teacher scene differs from the frozen component scene: {tscene} vs {scene}')
out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
report = train(scene, traj, out / 'training', centers=centers, feedback_gain=fg, velocity_gain=vg, seed=a.seed)
ck = out / 'training/policy.npz'
if decimals is not None:
    data = dict(np.load(ck, allow_pickle=False)); data['component_output_decimals'] = np.array(decimals); np.savez_compressed(ck, **data); report['component_output_decimals'] = decimals
report.update(teacher=str(teacher), teacher_trajectory_sha256=sha(traj), checkpoint_sha256=sha(ck)); json.dump(report, open(out / 'training/training_report.json', 'w'), indent=2)
cmd = [sys.executable, str(CODE / 'evaluate_policy_component.py'), '--episode', str(e), '--episode-dir', episode_dir, '--scene', scene, '--checkpoint', str(ck), '--output', str(out / 'evaluation')]
subprocess.run(cmd, check=True, stdout=open(out / 'evaluation.log', 'w'), stderr=subprocess.STDOUT)
r = json.load(open(out / 'evaluation/result.json')); m = r['metrics']['official_style_reference_proxy']
print(json.dumps(dict(episode=e, pass_=m['mp_sr'] == 1, fg=fg, centers=centers, objects={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in m['objects'].items()}, checkpoint=str(ck))))
