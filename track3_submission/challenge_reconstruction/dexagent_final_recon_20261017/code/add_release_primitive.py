"""Wrap a distilled LearnedPolicy checkpoint with the registered release primitive (release_feedback_policy.ReleasePolicy
over pinky_margin_policy_v2 with no margins), copying the release_* parameters from a donor checkpoint with optional
overrides, then evaluate with the UNCHANGED frozen evaluator.
  add_release_primitive.py --episode 3 --base <distilled policy.npz> --donor <old ep3 policy.npz> --output DIR
      [--start S] [--open-s X] [--place-s X] [--retreat-s X] [--place-scale F] [--retreat-scale F] [--open-scale F] [--decimals 6]
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; CODE = ROOT / 'code'; sys.path.insert(0, str(CODE))
from learned_feedback_policy import sha
ap = argparse.ArgumentParser(); ap.add_argument('--episode', type=int, required=True); ap.add_argument('--base', required=True); ap.add_argument('--donor', required=True); ap.add_argument('--output', required=True)
ap.add_argument('--start', type=float); ap.add_argument('--open-s', type=float); ap.add_argument('--place-s', type=float); ap.add_argument('--retreat-s', type=float)
ap.add_argument('--place-scale', type=float, default=1.0); ap.add_argument('--retreat-scale', type=float, default=1.0); ap.add_argument('--open-scale', type=float, default=1.0); ap.add_argument('--decimals', type=int, default=6)
ap.add_argument('--no-rotation', action='store_true'); ap.add_argument('--no-closed-bias', action='store_true')
a = ap.parse_args(); e = a.episode
base = dict(np.load(a.base, allow_pickle=False)); donor = np.load(a.donor, allow_pickle=False)
if str(base.get('algorithm')) != 'radial_basis_behavior_cloning_with_learned_feedback': raise SystemExit('base must be a plain LearnedPolicy checkpoint')
data = {k: v for k, v in base.items() if not k.startswith(('release', 'margin'))}
data.update(margin_base_module=np.array('learned_feedback_policy'), margin_base_class=np.array('LearnedPolicy'), margin_actuator_ids=np.zeros(0, int), margin_bias_rad=np.zeros(0), margin_startup_only=np.zeros(0, bool), margin_startup_end_s=np.array(0.8))
for k in donor.files:
    if k.startswith('release'): data[k] = donor[k].copy()
if a.no_rotation and 'release_place_rotation_vector' in data: del data['release_place_rotation_vector']
if a.no_closed_bias and 'release_closed_bias_rad' in data: del data['release_closed_bias_rad']
if a.start is not None: data['release_start_s'] = np.array(a.start)
if a.open_s is not None: data['release_open_duration_s'] = np.array(a.open_s)
if a.place_s is not None: data['release_place_duration_s'] = np.array(a.place_s)
if a.retreat_s is not None: data['release_retreat_duration_s'] = np.array(a.retreat_s)
data['release_place_delta_m'] = data['release_place_delta_m'] * a.place_scale; data['release_retreat_m'] = data['release_retreat_m'] * a.retreat_scale; data['release_open_delta_rad'] = data['release_open_delta_rad'] * a.open_scale
data['component_output_decimals'] = np.array(a.decimals)
out = Path(a.output).resolve(); (out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **data)
orig = ROOT / f'refinement_20261018/shared_policy_handoff/episode_{e:06d}/portable/original'; ev = json.loads((orig / 'evaluation_result.json').read_text())
base_report = {}
for cand in (Path(a.base).parent / 'training_report.json',):
    if cand.exists(): base_report = json.loads(cand.read_text())
report = dict(base_report, method='distilled LearnedPolicy + registered release primitive (release_feedback_policy.ReleasePolicy) with parameters copied from the donor checkpoint and explicit overrides',
              base_checkpoint=str(Path(a.base).resolve()), base_checkpoint_sha256=sha(a.base), donor_checkpoint=str(Path(a.donor).resolve()), donor_checkpoint_sha256=sha(a.donor),
              overrides=vars(a), release_parameters={k: (data[k].tolist() if data[k].size <= 6 else list(data[k].shape)) for k in data if k.startswith('release')}, checkpoint=str(ck), checkpoint_sha256=sha(ck), learned_weights_unchanged=True)
json.dump(report, open(out / 'training/training_report.json', 'w'), indent=2)
cmd = [sys.executable, str(CODE / 'evaluate_policy_component.py'), '--episode', str(e), '--episode-dir', ev['input_dir'], '--scene', ev['execution']['scene'], '--checkpoint', str(ck), '--output', str(out / 'evaluation')]
subprocess.run(cmd, check=True, stdout=open(out / 'evaluation.log', 'w'), stderr=subprocess.STDOUT)
r = json.load(open(out / 'evaluation/result.json')); m = r['metrics']['official_style_reference_proxy']
print(json.dumps(dict(episode=e, pass_=m['mp_sr'] == 1, objects={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in m['objects'].items()}, out=str(out))))
