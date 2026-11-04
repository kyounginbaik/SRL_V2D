"""Stream Q phase 6: compose board tracks -> our mesh body frames (G from meshreg ICP of our slotted mesh onto the compose mesh,
near-identity candidate = slot onto slot) -> our world by ONE rigid transform: X0 = world_from_rgb[0] @ A_T_rect @ inv(cam0_T_cam[0])
(h2h camera chain at frame 0), then a table correction Tt (rotation + vertical shift) that puts the compose RESTING boards (f0-30)
flat on our physics table z=0 (resting on their own collision). Prints the diagnostics. Library + CLI: compose_to_world.py OUT.npz"""
import sys, json, numpy as np, trimesh, glob
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/code', '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_other/code']
import meshreg
from common import A_T_rect
Q = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25'); CT = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/episode_000025')
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/collision_recon_slots_v3/episode_000025')
NAMES = ['wooden_piece_1', 'wooden_piece_2']
def build():
    Wc = np.load(SRC / 'object_poses.npz')['world_from_rgb'].astype(float); C = np.load(CT / 'camera_motion.npz')['cam0_T_cam'].astype(float)
    X0 = Wc[0] @ A_T_rect(25) @ np.linalg.inv(C[0]); G = {}; info = {}
    for o in NAMES:
        a = trimesh.load(Q / f'meshes_slotted_C/{o}_visual_slotted.obj', force='mesh'); b = trimesh.load(CT / f'{o}_mesh_metric.obj', force='mesh')
        res = meshreg.register(a, b); near = min(res, key=lambda r: R.from_matrix(r[0][:3, :3]).magnitude())
        G[o] = near[0]; info[o] = dict(chamfer_mm=near[1] * 1000, best_chamfer_mm=res[0][1] * 1000, rot_deg=float(np.degrees(R.from_matrix(near[0][:3, :3]).magnitude())), t_cm=(near[0][:3, 3] * 100).tolist())
    T = np.stack([np.einsum('ij,tjk->tik', X0, np.load(CT / f'{o}_trajectory.npz')['T_world_obj'].astype(float)) @ G[o] for o in NAMES], 1)   # world_from_ourbody
    # table correction from the resting boards f0-30
    coll = {o: np.vstack([trimesh.load(f, force='mesh').vertices for f in sorted(glob.glob(str(Q / f'collision_C/{o}_cutpz/part_*.obj')))]) for o in NAMES}
    nrm = []
    for i, o in enumerate(NAMES):
        for k in range(0, 31): nrm.append(T[k, i, :3, 1] * np.sign(T[k, i, 2, 1]))      # body y = board normal
    n = np.mean(nrm, 0); n /= np.linalg.norm(n); ax = np.cross(n, [0, 0, 1.]); s = np.linalg.norm(ax); Rt = R.from_rotvec(ax / s * np.arctan2(s, n[2])).as_matrix() if s > 1e-12 else np.eye(3)
    Tt = np.eye(4); Tt[:3, :3] = Rt; T1 = np.einsum('ij,tojk->toik', Tt, T)
    lows = [min((coll[o] @ T1[k, i, :3, :3].T + T1[k, i, :3, 3])[:, 2].min() for k in range(0, 31)) for i, o in enumerate(NAMES)]
    lows_each = {o: float(np.median([(coll[o] @ T1[k, i, :3, :3].T + T1[k, i, :3, 3])[:, 2].min() for k in range(0, 31)])) for i, o in enumerate(NAMES)}
    dz = -np.mean(list(lows_each.values())); Tt[2, 3] = dz; Tfin = np.einsum('ij,tojk->toik', Tt, T)
    info.update(X0=X0.tolist(), table_tilt_deg=float(np.degrees(np.arctan2(s, n[2]))), table_dz_cm=float(dz * 100), resting_lowest_cm_before_shift={o: v * 100 for o, v in lows_each.items()}, Tt=Tt.tolist())
    tilts = {o: [float(np.degrees(np.arccos(abs(Tfin[k, i, 2, 1])))) for k in (0, 15, 30)] for i, o in enumerate(NAMES)}
    info['resting_face_tilt_after_deg'] = tilts
    info['resting_lowest_after_cm'] = {o: float(np.median([(coll[o] @ Tfin[k, i, :3, :3].T + Tfin[k, i, :3, 3])[:, 2].min() for k in range(0, 31)]) * 100) for i, o in enumerate(NAMES)}
    return Tfin, G, info
if __name__ == '__main__':
    T, G, info = build(); np.savez(sys.argv[1], world_from_object=T, G1=G[NAMES[0]], G2=G[NAMES[1]]); print(json.dumps({k: v for k, v in info.items() if k not in ('X0', 'Tt')}, indent=1))
