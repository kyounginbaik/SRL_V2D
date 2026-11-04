"""Left-hand power-grasp (pocket) proposals on the pot handle, close to the pot body, for physics CEM (episodes 3, 5, 6).

Uses the unmodified DexAgent grasp-pocket tools (H.grasp_pocket / H.pocket_pose / H.finger_targets). Execution starts
from the original scene keyframe; object targets and timing are never changed. Writes one spec per
(handle point, finger direction) for physics_pickup_search.py.
"""
import json, sys
from pathlib import Path
import numpy as np, mujoco
from scipy.spatial.transform import Rotation
B = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(B.parents[1] / 'third_party/DexAgent')); sys.path.insert(0, str(B / 'code'))
from tool_library.common import hand as H, tracking as TR
from wrist_reference import continuous_euler_xyz
from physics_pickup_search import dump

EPISODES = {  # lift start L (reference), set-down S, approach timing
    1: dict(start=0.6, hover=1.0, contact=1.4, closed=1.75, transport=2.0, release=15.0, horizon=11.6),   # grasp the handle once before the lid comes off and hold until the lid is back on
    3: dict(start=1.5, hover=2.0, contact=2.4, closed=2.75, transport=3.05, release=11.6, horizon=5.7),
    5: dict(start=0.1, hover=0.8, contact=1.25, closed=1.55, transport=1.8, release=12.65, horizon=4.3),
    6: dict(start=0.25, hover=0.85, contact=1.3, closed=1.65, transport=1.95, release=8.45, horizon=4.5),
}
POINTS = {'near': 0.115, 'mid': 0.14}   # handle point along the pot body z axis (handle runs z 0.10..0.235, body radius ~0.09)


def prepare(episode, out_root):
    orig = B / f'refinement_20261018/shared_policy_handoff/episode_{episode:06d}/portable/original'
    ev = json.loads((orig / 'evaluation_result.json').read_text()); btr = json.loads((orig / 'base_training_report.json').read_text())
    import os
    scene = ev['execution']['scene']; episode_dir = Path(os.environ.get('POT_EPISODE_DIR') or ev['input_dir']); teacher = Path(os.environ.get('POT_TEACHER') or Path(btr['demonstration']).parents[1])
    manifest = json.loads((episode_dir / 'input_manifest.json').read_text())
    m = mujoco.MjModel.from_xml_path(scene); rig = H.HandRig.build(m, 'robot_left_', 'left'); d = mujoco.MjData(m)
    ref = TR.load_reference(teacher / 'reference.npz', m); Q = ref['qpos']; fps = ref['fps']
    original = TR.load_reference(episode_dir / 'tracking_reference.npz', m)['qpos']
    ts = EPISODES[episode]; frame = round(ts['contact'] * fps); q = Q[frame].copy()
    d.qpos[:] = q; mujoco.mj_forward(m, d); obj = m.body('white_pot').id; op = d.xpos[obj].copy(); Rm = d.xmat[obj].reshape(3, 3).copy()
    pocket = H.grasp_pocket(m, d, rig, .85)
    wc = np.array([rig.wrist_qpos[k] for k in ('x', 'y', 'z', 'rx', 'ry', 'rz')]); fcols = list(rig.finger_qpos.values())
    closed = rig.finger_targets(.90); opened = rig.finger_targets(.20)
    h = Rm[:, 2]; up = np.array([0., 0., 1.]); palm = -(up - (up @ h) * h); palm /= np.linalg.norm(palm)   # palm faces down onto the handle
    side_dir = np.cross(h, up); side_dir /= np.linalg.norm(side_dir)
    out = Path(out_root) / f'episode_{episode:06d}'; out.mkdir(parents=True, exist_ok=True); specs = []
    for pname, z in POINTS.items():
        point_local = np.array([.015, .003, z]); point = op + Rm @ point_local
        for sname, sign in (('a', 1.), ('b', -1.)):
            position, euler = H.pocket_pose(pocket, point, palm, sign * side_dir)
            pose = q.copy(); pose[wc[:3]] = position
            pose[wc[3:]] = continuous_euler_xyz(Rotation.from_euler('XYZ', [euler]), q[wc[3:]])[0]
            pose[fcols] = [closed[k] for k in rig.finger_qpos]
            name = f'{pname}_{sname}'; anchor = out / f'anchor_{name}.npz'; np.savez_compressed(anchor, qpos=pose)
            spec = dict(scene=scene, reference=str(teacher / 'reference.npz'), original_reference=str(episode_dir / 'tracking_reference.npz'),
                        source=manifest['source'], geometry=str(episode_dir / 'metric_geometry.json'), manifest=str(episode_dir / 'input_manifest.json'),
                        object_name='white_pot', side='left', anchor_pose=str(anchor), open_fingers=[opened[k] for k in rig.finger_qpos],
                        finger_names=list(rig.finger_qpos), times={k: ts[k] for k in ('start', 'hover', 'contact', 'closed', 'transport', 'release')},
                        horizon_s=ts['horizon'], hover_offset=[0., 0., .10], handle_point_local=point_local.tolist(), palm_direction_world=palm.tolist(),
                        finger_direction_world=(sign * side_dir).tolist(), grasp_generator='unmodified DexAgent H.grasp_pocket/H.pocket_pose/H.finger_targets (power grasp on the handle near the pot body)',
                        physical_initial_state='unchanged original scene keyframe', previous_teacher=str(teacher),
                        scale=np.r_[[.012] * 3, [.15] * 3, [.02] * 3, [.12] * 3, [.15] * 22, [.20] * 22, [.015] * 3, [.10] * 3, [.10, .08, .12]].tolist())
            dump(out / f'{name}.json', spec); specs.append(str(out / f'{name}.json'))
            print(f'ep{episode} {name}: pocket at {point.round(3)} wrist pos {np.round(position,3)} handle axis {h.round(2)} palm {palm.round(2)}')
    return specs


if __name__ == '__main__':
    import os, sys as _s; eps = [int(x) for x in _s.argv[1:]] or [3, 5, 6]
    root = Path(os.environ['POT_SPEC_ROOT']) if os.environ.get('POT_SPEC_ROOT') else B / ('refinement_20261021_ep1/power_grasp/specs' if eps == [1] else 'refinement_20261020_pot/power_grasp/specs'); all_specs = []
    for e in eps: all_specs += prepare(e, root)
    json.dump(all_specs, open(root / 'all_specs.json', 'w'), indent=1); print(len(all_specs), 'specs')
