"""Distil a corrected (wrapper-controlled) physical rollout into a plain registered LearnedPolicy checkpoint
and evaluate it with the UNCHANGED frozen evaluator (evaluate_policy_component.py).

The wrapper only exists to produce a teacher rollout. The distilled actor is the registered
radial-basis phase/proprioception architecture, so it needs no new runtime module.
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; CODE = ROOT / 'code'
sys.path.insert(0, str(CODE))
from learned_feedback_policy import train, sha

ap = argparse.ArgumentParser(); ap.add_argument('--variant', required=True, help='variant evaluation dir containing result.json')
ap.add_argument('--output', required=True); ap.add_argument('--centers', type=int); ap.add_argument('--feedback-gain', type=float)
ap.add_argument('--velocity-gain', type=float); ap.add_argument('--decimals', type=int); ap.add_argument('--seed', type=int, default=43)
a = ap.parse_args()
res = json.load(open(Path(a.variant) / 'result.json')); e = int(res['episode'].split('_')[-1])
base_dir = ROOT / f'refinement_20261018/shared_policy_handoff/episode_{e:06d}/portable/original'
with np.load(base_dir / 'policy.npz', allow_pickle=False) as z:
    centers = a.centers or len(z['centers']); decimals = a.decimals if a.decimals is not None else (int(z['component_output_decimals']) if 'component_output_decimals' in z.files else None)
rep = {}
for f in ('base_training_report.json', 'training_report.json'):   # the base report carries the RBF trainer's gains
    if (base_dir / f).exists():
        rep = json.load(open(base_dir / f))
        if rep.get('feedback_teacher_position_gain') is not None: break
fg = a.feedback_gain or rep.get('feedback_teacher_position_gain', .03); vg = a.velocity_gain or rep.get('feedback_teacher_velocity_gain', .0005)
out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
scene = res['execution']['scene']; teacher = res['execution']['trajectory']
report = train(scene, teacher, out / 'training', centers=centers, feedback_gain=fg, velocity_gain=vg, seed=a.seed)
ck = out / 'training/policy.npz'
if decimals is not None:
    data = dict(np.load(ck, allow_pickle=False)); data['component_output_decimals'] = np.array(decimals)
    np.savez_compressed(ck, **data); report['component_output_decimals'] = decimals
report.update(teacher_variant=str(Path(a.variant).resolve()), teacher_result_sha256=sha(Path(a.variant) / 'result.json'), checkpoint_sha256=sha(ck))
json.dump(report, open(out / 'training/training_report.json', 'w'), indent=2)
cmd = [sys.executable, str(CODE / 'evaluate_policy_component.py'), '--episode', str(e), '--episode-dir', res['input_dir'], '--scene', scene,
       '--checkpoint', str(ck), '--output', str(out / 'evaluation')]
print(' '.join(cmd), flush=True); subprocess.run(cmd, check=True)
r = json.load(open(out / 'evaluation/result.json')); m = r['metrics']['official_style_reference_proxy']
print(json.dumps(dict(episode=e, pass_=m['mp_sr'] == 1, objects={k: (v['mean_position_error_cm'][0], v['mean_orientation_error_degrees'][0]) for k, v in m['objects'].items()}, checkpoint=str(ck)), indent=1))
