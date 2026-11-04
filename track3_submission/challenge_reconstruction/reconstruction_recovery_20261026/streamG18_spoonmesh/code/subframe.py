"""G18: frame of the SUBMITTED v28 spoon mesh relative to our 5fcd334c mesh, per spoon episode (x_sub = T x_ours).
Rigid ICP (kit _align_rigid, no scale) of our mesh onto the submitted mesh, checked against the pose-derived map
inv(W_sub(t)) W_target(t) (median over frames; component vs our target, so it carries tracking error)."""
import sys, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/code')
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g18lib import *
from g10lib import load_sub, load_target
PKG = '/data1/SRL_V2D/challenge_submissions_SRL/track3_v28_20261029/all_20_episodes'
TGT = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17'
bv, bf = load(SPOON); out = {}
for ep in EPS:
    names, subT, subM = load_sub(PKG, ep); si = names.index('wooden_spoon')
    refT, _ = load_target(TGT, ep, names)
    sv, sf = subM[si]
    same = bool(np.allclose(np.sort(sv[:2052], 0), np.sort(bv, 0), atol=1e-7)) if len(np.unique(sv, axis=0)) == len(np.unique(bv, axis=0)) else False
    rel = np.einsum('tij,tjk->tik', np.linalg.inv(subT[si][:len(refT[si])]), refT[si][:len(subT[si])])
    from scipy.spatial.transform import Rotation as Rot
    Rp = Rot.from_matrix(rel[:, :3, :3]).mean().as_matrix(); tp = np.median(rel[:, :3, 3], 0)
    a = M._sample_mesh_surface(sv, sf, 8000, 1); b = M._sample_mesh_surface(bv, bf, 8000, 2)
    R, t = M._align_rigid(a, b)
    from scipy.spatial import cKDTree
    d = 0.5 * (cKDTree(a).query(b @ R.T + t)[0].mean() + cKDTree(b @ R.T + t).query(a)[0].mean())
    out[str(ep)] = dict(identical_vertices=same, T_sub_from_ours=H if False else np.r_[np.c_[R, t], [[0, 0, 0, 1]]].tolist(), icp_symdist_mm=float(d * 1000),
                        icp_ang_deg=angle_deg(R), icp_t_mm=(t * 1000).round(2).tolist(),
                        pose_vs_icp_deg=angle_deg(Rp.T @ R), pose_t_mm=(tp * 1000).round(2).tolist())
    if same: out[str(ep)]['T_sub_from_ours'] = np.eye(4).tolist()
    print(ep, {k: v for k, v in out[str(ep)].items() if k != 'T_sub_from_ours'}, flush=True)
json.dump(out, open(G / 'out/subframe.json', 'w'), indent=1)
