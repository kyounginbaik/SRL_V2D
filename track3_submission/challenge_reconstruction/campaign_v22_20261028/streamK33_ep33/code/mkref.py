"""E31: new episode dir for the stager / evaluator proxy gate whose object reference = the E14 target in the physics world.
Copies the actor's own world episode dir (input_dir of its evaluation; scene, robot reference, camera, setup untouched) into
streamK33_ep33/worlds/<V>/ep<EP>/episode_corrected/<episode> and replaces ONLY the object columns of reference_trajectory.npz
(object_positions / object_quaternions_wxyz / qpos object columns) and tracking_reference.npz (qpos object columns), every frame,
by mlibH.target_phys (target poses converted into the shipped mesh body frames by the kit registration and into the physics world by the
scorer's frame-0 alignment A of the actor's rollout) - the same mapping the beam's ILC uses. usage: E31_TGT=... mkref.py EP ACTOR MESHES V"""
import os, sys, json, shutil, hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mresH as R
from scipy.spatial.transform import Rotation as Rot
ep, actor, meshes, V = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]; name = f'episode_{ep:06d}'
env = R.Env(dict(ep=ep, actor=actor, meshes=meshes)); rep = env.evaluate(env.table0, keep=True)
src = env.epd; out = Path(__file__).resolve().parents[1] / 'worlds' / V / f'ep{ep}' / 'episode_corrected' / name
assert not out.exists(), out; shutil.copytree(src, out, symlinks=True)
tp = R.L.target_phys(ep, env.reg, rep['A'], env.tgt); rec = dict(stream='K33', episode=name, source_episode_dir=str(src), target=str(R.L.TGT), actor=actor, meshes=meshes,
    actor_rollout_objects=rep['objects'], A_physics_to_target=rep['A'].tolist(), replaced={})
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for fn in ('reference_trajectory.npz', 'tracking_reference.npz'):
    p = out / fn; z = np.load(p, allow_pickle=False); d = {k: z[k] for k in z.files}; old = sha(p)
    jn = [str(x) for x in d['joint_names']]; N = len(d['qpos'])
    for i, n in enumerate(env.names):
        T = tp[n][:N].copy()
        ax, order = R.L.K.symmetry_axis(n, R.L.geometry(ep)[n]['v']); ax = np.asarray(ax, float)
        if np.linalg.norm(ax) > 0 and int(order) != 1 and 'object_quaternions_wxyz' in d:
            # symmetric object: spin (scoring-invariant) about its symmetry axis, expressed in OUR body frame, chosen so that frame 0 matches the old reference
            axb = env.reg[n][1].T @ ax; axb /= np.linalg.norm(axb); j = [str(x) for x in d['object_names']].index(n)
            Rold0 = Rot.from_quat(d['object_quaternions_wxyz'][0, j], scalar_first=True); Rnew0 = Rot.from_matrix(T[0, :3, :3])
            best = min(np.linspace(-np.pi, np.pi, 3601), key=lambda th: (Rold0.inv() * Rnew0 * Rot.from_rotvec(th * axb)).magnitude())
            S = Rot.from_rotvec(best * axb).as_matrix(); T[:, :3, :3] = T[:, :3, :3] @ S; rec.setdefault('symmetric_spin_deg', {})[n] = float(np.degrees(best))
        pos = T[:, :3, 3]; q = Rot.from_matrix(T[:, :3, :3]).as_quat(scalar_first=True)
        k0 = jn.index(f'{n}_free') if f'{n}_free' in jn else None
        a = int(env.m.joint(f'{n}_free').qposadr[0])
        d['qpos'][:, a:a + 3] = pos; d['qpos'][:, a + 3:a + 7] = q
        if 'object_positions' in d:
            j = [str(x) for x in d['object_names']].index(n); d['object_positions'][:, j] = pos; d['object_quaternions_wxyz'][:, j] = q
    os.unlink(p); np.savez(p, **d) if fn == 'tracking_reference.npz' else np.savez(p, **d)
    rec['replaced'][fn] = dict(old_sha256=old, new_sha256=sha(p), frames=N, objects=env.names)
# the evaluator's proxy reference is manifest['source']/object_poses.npz (source frame; physics = inv(source_from_setup) @ source, exact for the
# Stream B worlds): copy the source dir, replace world_from_object of the target objects (all frames) by source_from_setup @ physics reference,
# and point the copied manifest's source / source_object_poses_sha256 at it (every other manifest field unchanged)
man = json.loads((out / 'input_manifest.json').read_text()); ssrc = Path(man['source']); SfS = np.asarray(man['source_from_setup'], float)
sdst = out.parents[1] / 'source_corrected' / name; assert not sdst.exists(); shutil.copytree(ssrc, sdst, symlinks=False)
z = np.load(sdst / 'object_poses.npz', allow_pickle=False); d = {k: z[k] for k in z.files}; old = sha(sdst / 'object_poses.npz'); snames = [str(x) for x in d['names']]
rt = np.load(out / 'reference_trajectory.npz', allow_pickle=False); rn = [str(x) for x in rt['object_names']]
for n in env.names:
    j = rn.index(n); P = np.tile(np.eye(4), (len(rt['object_positions']), 1, 1)); P[:, :3, :3] = Rot.from_quat(rt['object_quaternions_wxyz'][:, j], scalar_first=True).as_matrix(); P[:, :3, 3] = rt['object_positions'][:, j]
    d['world_from_object'][:, snames.index(n)] = (SfS @ P).astype(d['world_from_object'].dtype)
os.unlink(sdst / 'object_poses.npz'); np.savez_compressed(sdst / 'object_poses.npz', **d)
man['source'] = str(sdst); man['source_object_poses_sha256'] = sha(sdst / 'object_poses.npz'); os.unlink(out / 'input_manifest.json'); (out / 'input_manifest.json').write_text(json.dumps(man, indent=2) + '\n')
rec['source_copy'] = dict(source=str(ssrc), copy=str(sdst), old_object_poses_sha256=old, new_object_poses_sha256=man['source_object_poses_sha256'])
(out / 'e31_reference.json').write_text(json.dumps(rec, indent=1))
print('REF_OK', out, json.dumps(rec['replaced']), rep['objects'])
