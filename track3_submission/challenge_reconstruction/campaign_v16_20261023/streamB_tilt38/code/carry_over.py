"""Carry a teacher over to another setup frame of the same episode (or replay it in place as a control).
The wrist trajectories (position + XYZ Euler, continuous branch) are transformed by T = setup_from_source_new @ inv(setup_from_source_old);
fingers unchanged; the scene is _freeze(stage2/scene.xml) + controller_scene(variant) of the NEW episode dir (identical recipe to the old
teachers, verified at MjModel level for ep3). Output dir gets reference.npz, execution/, result.json (ILC / distill compatible).
usage: carry_over.py --episode E --old-epdir D --new-epdir D --out DIR (--reference REF.npz | --ctrl-trajectory TRAJ.npz) [--variant kp5000_r200_d5]"""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[k] = '1'
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation as R
R0 = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(R0 / 'code'), '/data1/SRL_V2D/third_party/DexAgent']
from floating_runtime import _freeze, execute_reference
from calibrate_controller import controller_scene
from evaluate_mp_sr import evaluate_files
from wrist_reference import continuous_euler_xyz
from tool_library.common import tracking as TR
from tool_library.common import episode as E
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
W = ['pos_x', 'pos_y', 'pos_z', 'rot_x', 'rot_y', 'rot_z']

def build_scene(epdir, out, variant):
    out.mkdir(parents=True, exist_ok=True); m, frozen = _freeze(Path(epdir) / 'stage2/scene.xml', out)
    scene = out / 'controller_scene.xml'; controller_scene(frozen, scene, variant); return scene

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--episode', type=int, required=True); ap.add_argument('--old-epdir', required=True); ap.add_argument('--new-epdir', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--reference'); ap.add_argument('--ctrl-trajectory'); ap.add_argument('--variant', default='kp5000_r200_d5'); ap.add_argument('--scene', help='reuse an existing controller scene'); ap.add_argument('--no-clip', action='store_true', help='do not clip commands to joint ranges (servo clips to ctrlrange anyway)')
    a = ap.parse_args(); out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True); old, new = Path(a.old_epdir), Path(a.new_epdir)
    S_old = np.array(json.loads((old / 'input_manifest.json').read_text())['setup_from_source']); S_new = np.array(json.loads((new / 'input_manifest.json').read_text())['setup_from_source'])
    T = S_new @ np.linalg.inv(S_old); Rt, tt = T[:3, :3], T[:3, 3]
    scene = Path(a.scene) if a.scene else build_scene(new, out / 'scene', a.variant)
    m = mujoco.MjModel.from_xml_path(str(scene)); names = E.joint_names(m)
    new_ref = TR.load_reference(new / 'tracking_reference.npz', m); Qnew = new_ref['qpos']
    if a.reference:
        ref = TR.load_reference(a.reference, m); Q = ref['qpos'].copy(); labels = ref['labels']; fps = ref['fps']; seg, segs = ref['segment'], ref['segments']; source = f'carry-over of {a.reference}'
    else:
        tr = np.load(a.ctrl_trajectory); act = [str(x) for x in tr['actuator_names']]; assert act == [m.actuator(i).name for i in range(m.nu)]
        qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]; C = tr['ctrl']; Tn = len(C); Q = np.tile(Qnew[0][None], (Tn, 1)); Q[:, qadr] = C
        # frame 0 = home state (interpolation start), later frames = commanded targets of that frame
        Q[0, qadr] = Qnew[0, qadr]; fps = float(tr['fps']); labels = ['both: reference tracking'] * Tn; seg = np.zeros(Tn, int); segs = [{'stage1_indices': [0], 'motion': 'track_reference', 'side': 'both'}]; source = f'carry-over of commanded ctrl from {a.ctrl_trajectory}'
    Tn = len(Q)
    for side in ('left', 'right'):
        cols = [names.index(f'robot_{side}_{side}_{w}') for w in W]
        p = Q[:, cols[:3]]; ang = Q[:, cols[3:]]
        Q[:, cols[:3]] = p @ Rt.T + tt
        rot = R.from_matrix(Rt) * R.from_euler('XYZ', ang)
        Q[:, cols[3:]] = continuous_euler_xyz(rot, Qnew[0, cols[3:]])
        # consistency: transformed frame-0 wrist must coincide with the new episode's own frame-0 wrist (both derive from the same source row)
        d_pos = float(np.linalg.norm(Q[0, cols[:3]] - Qnew[0, cols[:3]])); d_rot = float(np.degrees((R.from_euler('XYZ', Q[0, cols[3:]]) * R.from_euler('XYZ', Qnew[0, cols[3:]]).inv()).magnitude()))
        print(json.dumps(dict(side=side, frame0_wrist_pos_mismatch_m=d_pos, frame0_wrist_rot_mismatch_deg=d_rot)), flush=True)
    # object columns: new episode's own targets (never commanded); frame 0 of robot columns = exact new home
    for n in [str(x) for x in new_ref.get('object_names', [])] if 'object_names' in new_ref else []:
        pass
    obj_cols = [i for i, n in enumerate(names) if not n.startswith('robot_')]
    Q[:min(Tn, len(Qnew)), obj_cols] = Qnew[:min(Tn, len(Qnew)), obj_cols]
    lim = m.jnt_range[m.actuator_trnid[:, 0]]; qa = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    if not a.no_clip: Q[:, qa] = np.clip(Q[:, qa], lim[:, 0], lim[:, 1])
    Q[0] = Qnew[0]
    refp = TR.save_reference(out / 'reference.npz', m, Q, list(labels), fps, seg, segs, source=source, arms_coupled=False, floating_runtime=True)
    ex = execute_reference(str(scene), refp, out / 'execution'); man = json.loads((new / 'input_manifest.json').read_text())
    met = evaluate_files(man['source'], ex['trajectory'], new / 'input_manifest.json', ex['scene'], new / 'metric_geometry.json')
    res = dict(episode=f'episode_{a.episode:06d}', input_dir=str(new), reference=str(refp), execution=ex, metrics=met, learned_policy=False,
               carry_over=dict(old_epdir=str(old), new_epdir=str(new), setup_change=T.tolist(), source=a.reference or a.ctrl_trajectory, variant=a.variant, scene=str(scene), scene_sha256=sha(scene)),
               spec=dict(episode=f'episode_{a.episode:06d}', scene=str(scene), reference=str(refp), source=man['source'], manifest=str(new / 'input_manifest.json'), geometry=str(new / 'metric_geometry.json'), original_reference=str(new / 'tracking_reference.npz')))
    (out / 'result.json').write_text(json.dumps(res, indent=1, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
    o = {k: (round(v['mean_position_error_cm'][0], 2), round(v['mean_orientation_error_degrees'][0], 1)) for k, v in met['official_style_reference_proxy']['objects'].items()}
    print(json.dumps(dict(out=str(out), mp_sr=met['reference_proxy_mp_sr'], objects=o, completed=ex['execution_completed'], physics_verified=ex['physics_verified'])), flush=True)

if __name__ == '__main__': main()
