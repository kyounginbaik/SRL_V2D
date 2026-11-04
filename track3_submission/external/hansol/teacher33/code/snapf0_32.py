"""R32: copy of an mkref episode world whose reference frame 0 object columns are snapped to the scene's initial (keyframe) state.
The team margin search (refine_grasp_cem.Experiment) requires reference frame 0 == scene initial state within 1e-5; mkref maps our target
through a rollout's frame-0 alignment, which leaves a small frame-0 gap (W1: spoon 6.7 mm, cup 0.7 mm). Only frame 0 of the object columns
(qpos, and object_positions / object_quaternions_wxyz if present) changes; every other frame and every robot column is byte-identical.
usage: snapf0_32.py SRC_EPISODE_DIR DST_EPISODE_DIR SCENE_XML"""
import sys, json, shutil, hashlib, numpy as np, mujoco
from pathlib import Path
src, dst, scene = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
assert not dst.exists(), dst; shutil.copytree(src, dst)
m = mujoco.MjModel.from_xml_path(scene); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
names = ['blue_cup', 'wooden_spoon']; adr = {n: int(m.joint(f'{n}_free').qposadr[0]) for n in names}; rec = {}
for f in ('tracking_reference.npz', 'reference_trajectory.npz'):
    p = dst / f; z = np.load(p, allow_pickle=False); a = {k: z[k] for k in z.files}; before = hashlib.sha256(p.read_bytes()).hexdigest()
    for n in names:
        i = adr[n]; gap = float(1000 * np.linalg.norm(a['qpos'][0, i:i + 3] - d.qpos[i:i + 3])); a['qpos'][0, i:i + 7] = d.qpos[i:i + 7]
        if 'object_positions' in a:
            k = [str(x) for x in a['object_names']].index(n); a['object_positions'][0, k] = d.qpos[i:i + 3]; a['object_quaternions_wxyz'][0, k] = d.qpos[i + 3:i + 7]
        rec[f'{f}:{n}'] = dict(frame0_gap_mm=round(gap, 3))
    p.unlink(); np.savez_compressed(p, **a); rec[f] = dict(old_sha256=before, new_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(dst / 'R32_SNAPF0.json').write_text(json.dumps(dict(source=str(src), scene=scene, rule='frame 0 object columns := scene keyframe state; all else unchanged', changes=rec), indent=1))
print('SNAP_OK', json.dumps(rec))
