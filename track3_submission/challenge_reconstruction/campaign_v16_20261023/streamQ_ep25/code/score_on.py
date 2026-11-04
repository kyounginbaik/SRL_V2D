"""Re-score an executed trajectory (trajectory.npz) against a given reference object_poses.npz with the unchanged official-style proxy
(evaluate_mp_sr.evaluate_arrays, geometry of the original slots_v3 source). Prints cm/deg per object, alignment rotation, and optional
1-s bins. usage: score_on.py TRAJ.npz POSES.npz [--bins] [--first-frame-from TRAJ2|none]"""
import sys, json, numpy as np
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
import evaluate_mp_sr as M
from scipy.spatial.transform import Rotation
SRC = '/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3/adapter_sources/episode_000025'
def ref_from(poses_path):
    z = np.load(poses_path); P = np.asarray(z['world_from_object'], float); names = z['names']
    q = Rotation.from_matrix(P[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(*P.shape[:2], 4)
    return M.validate_poses(dict(timestamps_seconds=np.arange(len(P)) / 20., object_positions=P[:, None, :, :3, 3], object_quaternions_wxyz=q[:, None][..., [3, 0, 1, 2]], object_names=names))
def score(traj, poses, bins=False):
    geo = M.geometry_from_reference(SRC); ref = ref_from(poses); tr = M.load_trajectory(traj)
    rep = M.evaluate_arrays(ref, tr, geo)
    o = rep['official_style_reference_proxy']['objects']; T = np.array(rep['official_initial_scene_transforms'][0])
    out = dict(mp_sr=rep['reference_proxy_mp_sr'], align_rot_deg=float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))),
               objects={k: (round(v['mean_position_error_cm'][0], 3), round(v['mean_orientation_error_degrees'][0], 2), round(v['max_position_error_m'][0] * 100, 2)) for k, v in o.items()})
    if bins:   # per-1s bins (aligned)
        ref = ref_from(poses); tr = M.resample_to_reference(M.load_trajectory(traj), ref); anchors, axes, kinds = M.validate_geometry(ref['object_names'], geo)
        for d in (ref, tr): d['object_positions'] = d['object_positions'] + np.einsum('twoij,oj->twoi', M.rotation_matrices(d['object_quaternions_wxyz']), anchors)
        reg, _ = M.align_initial_scene(tr, ref, axes, kinds)
        pe = np.linalg.norm(reg['object_positions'] - ref['object_positions'], axis=-1)[:, 0]; ae = np.degrees(M.angular_errors(reg['object_quaternions_wxyz'], ref['object_quaternions_wxyz'], axes, kinds))[:, 0]
        out['bins'] = [[s, *np.round(pe[s * 20:(s + 1) * 20].mean(0) * 100, 2).tolist(), *np.round(ae[s * 20:(s + 1) * 20].mean(0), 1).tolist()] for s in range(int(np.ceil(len(pe) / 20)))]
    return out
if __name__ == '__main__':
    r = score(sys.argv[1], sys.argv[2], '--bins' in sys.argv)
    print(json.dumps({k: v for k, v in r.items() if k != 'bins'}))
    for b in r.get('bins', []): print('  %2ds  p1 %5.2f cm  p2 %5.2f cm | p1 %5.1f deg  p2 %5.1f deg' % tuple(b))
