"""Stream N copy for the wooden_spoon (object joint renamed only). Episode 30 release variant (v9). The previous release froze the wrist only from 12.3 s, but from ~11.8 s the human-derived
wrist reference already pulls away and drags the brush (20 N hand force spike, brush knocked 6 cm and tipped to 60 deg).
Here the wrist is frozen from RELEASE (the brush is in its slot at ~11.6-12.0 s), optionally lowered by LOWER over 0.3 s before
the release so the head seats between the posts, fingers open to neutral over OPEN_S, then the wrist retreats straight UP
first and sideways afterwards. Object reference unchanged.
usage: release_v9.py SOURCE_RUN EPISODE_DIR OUT RELEASE OPEN_S LOWER UP LATERAL [MODE neutral|body|pinch] [APERTURE_M]"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import sys, json
from pathlib import Path
import numpy as np, mujoco
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017')
sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import hand as H, tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
src, inp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); release, open_s, lower, up, lateral = map(float, sys.argv[4:9]); mode = sys.argv[9] if len(sys.argv) > 9 else 'neutral'; aperture = float(sys.argv[10]) if len(sys.argv) > 10 else .02
b = lambda x: np.clip(x, 0, 1) ** 3 * (10 - 15 * np.clip(x, 0, 1) + 6 * np.clip(x, 0, 1) ** 2)
rep = json.loads((src / 'result.json').read_text()); man = json.loads((inp / 'input_manifest.json').read_text()); scene = rep['execution']['scene']
m = mujoco.MjModel.from_xml_path(scene); ref = TR.load_reference(rep['reference'], m); Q = ref['qpos'].copy(); orig = Q.copy(); fps = float(ref['fps']); t = np.arange(len(Q)) / fps; k = round(release * fps)
rig = H.HandRig.build(m, 'robot_right_', 'right'); wc = np.array([rig.wrist_qpos[j] for j in ['x', 'y', 'z', 'rx', 'ry', 'rz']]); fc = np.array(list(rig.finger_qpos.values())); cols = np.r_[wc, fc]
oc = int(m.joint('wooden_spoon_free').qposadr[0])
Q[np.ix_(t >= release, cols)] = Q[k, cols]                                   # freeze wrist + fingers at the release pose
Q[:, wc[2]] -= lower * b((t - (release - .3)) / .3)                          # seat the head a little lower before opening
ids = m.actuator_trnid[list(rig.finger_act.values()), 0]
if mode == 'neutral': opened = np.clip(np.zeros(len(fc)), m.jnt_range[ids, 0], m.jnt_range[ids, 1])
else:   # fingertips move radially away from the brush body (or the thumb-index pinch point) by APERTURE
    d = mujoco.MjData(m); d.qpos[:] = Q[k]; mujoco.mj_forward(m, d); pinch = .5 * (d.site_xpos[rig.fingertip_sites['thumb']] + d.site_xpos[rig.fingertip_sites['index']]); targets = {}
    for finger in H.FINGERS:
        pos = d.site_xpos[rig.fingertip_sites[finger]].copy(); radial = pos - (pinch if mode == 'pinch' else Q[k, oc:oc + 3])
        if mode == 'body': radial[2] = 0
        radial /= max(np.linalg.norm(radial), 1e-8); targets[finger] = pos + aperture * radial
    H.finger_ik(m, d, rig, targets, iters=180); opened = d.qpos[fc].copy()
Q[:, fc] += (opened - Q[:, fc]) * b((t - release) / open_s)[:, None]
direction = Q[k, wc[:3]] - Q[k, oc:oc + 3]; direction[2] = 0; direction /= max(np.linalg.norm(direction), 1e-8)
Q[:, wc[2]] += up * b((t - release - open_s) / .5)
Q[:, wc[:3]] += lateral * direction * b((t - release - open_s - .5) / .6)[:, None]
other = np.setdiff1d(np.arange(m.nq), cols); assert np.array_equal(Q[:, other], orig[:, other]); assert np.array_equal(Q[t < release - .3], orig[t < release - .3])
robot = m.jnt_qposadr[m.actuator_trnid[:, 0]]; Q[:, robot] = np.clip(Q[:, robot], m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
out.mkdir(parents=True, exist_ok=False)
path = TR.save_reference(out / 'reference.npz', m, Q, ref['labels'], ref['fps'], ref['segment'], ref['segments'], source='v9 release: early wrist freeze, optional seat-lower, slow neutral opening, up-then-lateral retreat; unchanged object reference', floating_runtime=True)
ex = execute_reference(scene, path, out / 'execution'); met = evaluate_files(man['source'], ex['trajectory'], inp / 'input_manifest.json', ex['scene'], inp / 'metric_geometry.json')
res = dict(episode=inp.name, input_dir=str(inp), reference=str(path), execution=ex, metrics=met, release_seconds=release, open_s=open_s, opening_mode=mode, aperture_m=aperture, lower_m=lower, up_m=up, lateral_m=lateral, parent=str(src), object_targets_unchanged=True, learned_policy=False)
(out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects={kk: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for kk, v in met['official_style_reference_proxy']['objects'].items()})), flush=True)
