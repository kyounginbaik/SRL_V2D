"""Episode 33: zero-margin release wrapper (the v11 recipe): actor (LearnedPolicy) -> PinkyMarginPolicy with ZERO margins on the
right-hand finger actuators -> MultiReleasePolicy with the shipped release keys (left 15.8 s, right 13.9 s + placement).
Evaluated with the unchanged evaluate_policy_component. usage: wrap_zero_release33.py ACTOR_DIR OUT_DIR"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json, hashlib
from pathlib import Path
import numpy as np, mujoco
B = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(B / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
DONOR = B / 'refinement_20261018/expanded_batch/bimanual_release/episode_000033/case_0/policy.npz'
actor, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(); sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
ev = json.loads((actor / 'evaluation/result.json').read_text()); scene = ev['execution']['scene']; m = mujoco.MjModel.from_xml_path(scene)
with np.load(actor / 'training/policy.npz', allow_pickle=False) as z: d = {k: z[k] for k in z.files}
ids = [int(np.flatnonzero(m.actuator_trnid[:, 0] == j)[0]) for j in range(m.njnt) if (m.joint(j).name or '').startswith('robot_right_right_') and any(f'_{f}_' in m.joint(j).name for f in ['thumb', 'index', 'middle', 'ring', 'pinky']) and len(np.flatnonzero(m.actuator_trnid[:, 0] == j)) == 1]
d.update(margin_base_module=np.array('learned_feedback_policy'), margin_base_class=np.array('LearnedPolicy'), margin_actuator_ids=np.asarray(ids, int), margin_bias_rad=np.zeros(len(ids)),
         margin_startup_only=np.zeros(len(ids), bool), margin_startup_end_s=np.array(.8))
with np.load(DONOR, allow_pickle=False) as z:
    for k in z.files:
        if k.startswith('release') or k.startswith('multi_release'): d[k] = z[k]
(out / 'training').mkdir(parents=True, exist_ok=True); ck = out / 'training/policy.npz'; np.savez_compressed(ck, **d)
tr = json.loads((actor / 'training/training_report.json').read_text())
tr.update(release_wrapper=dict(recipe='zero right-finger margins + shipped MultiReleasePolicy release keys', donor=str(DONOR), donor_sha256=sha(DONOR), base_actor=str(actor / 'training/policy.npz'),
          base_actor_sha256=sha(actor / 'training/policy.npz'), wrapper_script=str(Path(__file__).resolve()), wrapper_sha256=sha(__file__)), checkpoint_sha256=sha(ck))
(out / 'training/training_report.json').write_text(json.dumps(tr, indent=2, default=str))
from evaluate_policy_component import evaluate
evaluate(33, scene, str(ck), str(out / 'evaluation'), ev['input_dir'])
r = json.loads((out / 'evaluation/result.json').read_text()); mm = r['metrics']['official_style_reference_proxy']['objects']
print(json.dumps(dict(out=str(out), passed=r['metrics']['reference_proxy_mp_sr'] == 1 and r['execution']['physics_verified'], objects={k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in mm.items()})), flush=True)
