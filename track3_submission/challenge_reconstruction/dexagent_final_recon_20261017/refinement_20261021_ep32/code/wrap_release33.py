"""Episode 33: graft the shipped MultiReleasePolicy release/margin keys (both hands release at 13.9 / 15.8 s, pinky startup
margin) onto a newly distilled LearnedPolicy actor of the same scene; evaluate with the unchanged evaluate_policy_component.
usage: wrap_release33.py ACTOR_DIR OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, hashlib, subprocess, shutil
from pathlib import Path
import numpy as np
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
DONOR = B / 'refinement_20261018/expanded_batch/bimanual_release/episode_000033/case_0/policy.npz'
BASE_KEYS = ['algorithm', 'centers', 'width', 'phase_weights', 'feedback_weights', 'state_mean', 'state_scale', 'action_mean', 'action_scale', 'qadr', 'vadr', 'ctrl_lo', 'ctrl_hi', 'actuator_names', 'duration_s', 'fps', 'frames', 'scene_sha256']
actor, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
donor = dict(np.load(DONOR, allow_pickle=False)); new = dict(np.load(actor / 'training/policy.npz', allow_pickle=False))
if str(donor['scene_sha256']) != str(new['scene_sha256']): raise SystemExit('scene mismatch')
for k in BASE_KEYS: donor[k] = new[k]
for k in new:
    if k not in BASE_KEYS: donor[k] = new[k]
(out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **donor)
rep = json.loads((actor / 'training/training_report.json').read_text())
rep.update(release_wrapper=dict(donor=str(DONOR), donor_sha256=sha(DONOR), base_actor=str(actor / 'training/policy.npz'), base_actor_sha256=sha(actor / 'training/policy.npz'),
           architecture='multi_release_feedback_policy.MultiReleasePolicy over pinky_margin_policy_v2.PinkyMarginPolicy over learned_feedback_policy.LearnedPolicy'), checkpoint_sha256=sha(ck))
(out / 'training/training_report.json').write_text(json.dumps(rep, indent=2))
ev = json.loads((actor / 'evaluation/result.json').read_text())
cmd = [sys.executable, str(B / 'code/evaluate_policy_component.py'), '--episode', '33', '--episode-dir', ev['input_dir'], '--scene', ev['execution']['scene'] if 'scene' in ev['execution'] else '', '--checkpoint', str(ck), '--output', str(out / 'evaluation')]
subprocess.run(cmd, check=True, stdout=open(out / 'evaluation.log', 'w'), stderr=subprocess.STDOUT, cwd='/tmp')
r = json.loads((out / 'evaluation/result.json').read_text()); m = r['metrics']['official_style_reference_proxy']
print(json.dumps(dict(actor=actor.name, src=str(actor.parent.parent.name), passed=r['metrics']['reference_proxy_mp_sr'] == 1 and r['execution']['physics_verified'], arch=r.get('policy_architecture'),
                      objects={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in m['objects'].items()})), flush=True)
