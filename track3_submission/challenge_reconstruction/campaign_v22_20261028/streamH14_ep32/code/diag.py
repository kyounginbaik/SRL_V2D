"""H12: per-frame spoon error decomposition (twist about the spoon long axis vs swing) of a table's rollout vs target_final_v12b_g15 (H14 copy). usage: diag.py ACTOR MESHES TABLE"""
import sys, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, str(Path(__file__).resolve().parent)); import mres32H13 as mres
env = mres.Env(dict(ep=32, actor=sys.argv[1], meshes=sys.argv[2], table=sys.argv[3] if len(sys.argv) > 3 else None))
r = env.evaluate(env.table0, keep=True); Q = r['Q']; e = r['err']; A = r['A']
tp = mres.L.target_phys(32, env.reg, A, env.tgt)['wooden_spoon']; P = env.poses(Q)['wooden_spoon']
vv = np.asarray(trimesh.load(Path(sys.argv[2]) / 'wooden_spoon.obj', force='mesh', process=False).vertices); vv -= vv.mean(0); ax = np.linalg.svd(vv, full_matrices=False)[2][0]
c = np.load('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG12/out/a32_consensus.npz')['consensus']
print('objects', r['objects'])
tw_all = []
for k in range(env.T):
    Rrel = tp[k, :3, :3].T @ P[k, :3, :3]; rv = R.from_matrix(Rrel).as_rotvec(); q = R.from_matrix(Rrel).as_quat()  # xyzw
    proj = np.dot(q[:3], ax); tw = np.degrees(2 * np.arctan2(proj, q[3])); tw = (tw + 180) % 360 - 180
    sw = R.from_matrix(Rrel) * R.from_rotvec(-np.radians(tw) * ax); tw_all.append(tw)
    if k % int(__import__("os").environ.get("EVERY", 4)) == 0: print(f'f{k:3d} tot {e["wooden_spoon"][1][k]:6.1f} twist {tw:7.1f} swing {np.degrees(sw.magnitude()):6.1f} pos {e["wooden_spoon"][0][k]:5.2f} tgt_roll {c[k]:6.0f}')
tw_all = np.array(tw_all); print('mean |twist|', np.abs(tw_all).mean().round(1))
