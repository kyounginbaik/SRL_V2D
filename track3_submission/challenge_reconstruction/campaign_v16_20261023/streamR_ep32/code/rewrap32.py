"""Stream R ep32: wrap the v19 actor (J_ep32_chainB: PinkyMarginPolicy over LearnedPolicy; weights and margins untouched) in the
existing release_feedback_policy.ReleasePolicy for the LEFT hand (cup): hold the learned left phase from START, optionally lower the
wrist by PLACE_DZ over PLACE_DUR (set the cup down instead of dropping it), open the fingers over OPEN_DUR by
OPEN_SCALE x (the actor's own final left finger command minus its command at START, read from the v19 rollout), then retreat
by RETREAT_Z over RETREAT_DUR. The right hand is not touched. Evaluated with the unchanged evaluate_policy_component.
usage: rewrap32.py OUT_DIR start=.. [place_dz=0 place_dur=0.4 open_dur=0.6 open_scale=1 retreat_z=0.05 retreat_dur=1.0]"""
import os, sys, json, hashlib
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
from pathlib import Path
import numpy as np
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(B / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from evaluate_policy_component import evaluate
BASE = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/ep32/margins/c2_c4096_cold')
ROLL = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/J_ep32_chainB/episode_000032/backend/evaluation/execution/trajectory.npz')
out = Path(sys.argv[1]).resolve(); p = dict(place_dz=0., place_dur=0.4, open_dur=0.6, open_scale=1., retreat_z=0.05, retreat_dur=1.0)
for kv in sys.argv[2:]: k, v = kv.split('='); p[k] = float(v)
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
ev = json.loads((BASE / 'evaluation/result.json').read_text()); scene = ev['execution']['scene']; epd = ev['input_dir']
with np.load(BASE / 'training/policy.npz', allow_pickle=False) as z: d = {k: z[k] for k in z.files}
names = [str(x) for x in d['actuator_names']]; L = [i for i, n in enumerate(names) if n.startswith('robot_left_left_')]
tr_ids = [names.index(f'robot_left_left_pos_{a}_position') for a in 'xyz']; rot_ids = [names.index(f'robot_left_left_rot_{a}_position') for a in 'xyz']
fing = [i for i in L if i not in tr_ids + rot_ids]
roll = np.load(ROLL); ctrl = roll['ctrl']; an = [str(x) for x in roll['actuator_names']]; assert an == names
k0 = int(round(p['start'] * 20)); delta = (ctrl[-1, fing] - ctrl[k0, fing]) * p['open_scale']
d.update(release_base_module=np.array('pinky_margin_policy_v2'), release_base_class=np.array('PinkyMarginPolicy'), release_start_s=np.array(p['start']),
         release_open_duration_s=np.array(p['open_dur']), release_retreat_duration_s=np.array(p['retreat_dur']), release_actuator_ids=np.array(L),
         release_finger_actuator_ids=np.array(fing), release_translation_actuator_ids=np.array(tr_ids), release_rotation_actuator_ids=np.array(rot_ids),
         release_open_delta_rad=delta, release_retreat_m=np.array([0., 0., p['retreat_z']]))
if p['place_dz']: d.update(release_place_duration_s=np.array(p['place_dur']), release_place_delta_m=np.array([0., 0., p['place_dz']]))
(out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **d)
tr = json.loads((BASE / 'training/training_report.json').read_text())
tr.update(left_release_wrapper=dict(base_actor=str(BASE / 'training/policy.npz'), base_sha256=sha(BASE / 'training/policy.npz'), parameters=p, open_delta_source=str(ROLL), open_delta_source_sha256=sha(ROLL),
          script=str(Path(__file__).resolve()), script_sha256=sha(__file__)), checkpoint_sha256=sha(ck))
(out / 'training/training_report.json').write_text(json.dumps(tr, indent=2, default=str))
evaluate(32, scene, str(ck), str(out / 'evaluation'), epd)
r = json.loads((out / 'evaluation/result.json').read_text()); m = r['metrics']['official_style_reference_proxy']
print('DONE', json.dumps(dict(params=p, mp_sr=m['mp_sr'], objects={k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 1)) for k, v in m['objects'].items()})), flush=True)
