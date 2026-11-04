"""C30 coworker-mesh alignment (user instruction 2026-11-01 night). For every episode/object of the coworker's
final_episode_meshes_20261029 (spoon excluded: no coworker spoon, keep G18), express their mesh in OUR pose body frame:
multi-start rigid ICP (no scaling) of their mesh onto our submitted v29 mesh for that episode/object; only the rigid transform is
applied to their vertices. Check: kit MP-SR _register_object_body of the new (kit-budgeted) mesh onto our mesh, kit seed for the slot
+ 9 more seeds; pass = every seed rotation < 5 deg (same twin). Also reports the best competing ICP basin (> 30 deg away).
Read-only on inputs; writes OUT/episode_XXXXXX/<obj>.obj + OUT/cwalign_report.json. usage: cwalign.py OUT"""
import sys, json, numpy as np, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
CW = Path('/data1/SRL_V2D/challenge_submissions_SRL/cowork_recon_gdrive_20261101/Jisu/final_episode_meshes_20261029')
OURS = Path('/data1/SRL_V2D/challenge_submissions_SRL/track3_v29_20261030/all_20_episodes/meshes')
B14 = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v12b_g15')   # names/slot order (== v14)
OUT = Path(sys.argv[1]); assert not OUT.exists(), OUT; OUT.mkdir(parents=True)
def load(p):
    m = trimesh.load(p, force='mesh', process=False); return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
def kab(P, Q):
    mp, mq = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((Q - mq).T @ (P - mp)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T; return R, mp - R @ mq
def pca(X): w, U = np.linalg.eigh(np.cov((X - X.mean(0)).T)); return U[:, ::-1]
def icp(P, Q, R, t, it=60):
    tp = cKDTree(P)
    for _ in range(it):
        q = Q @ R.T + t; d, i = tp.query(q); keep = d <= np.percentile(d, 95)
        R2, t2 = kab(P[i[keep]], q[keep]); R, t = R2 @ R, R2 @ t + t2
    q = Q @ R.T + t; d1 = tp.query(q)[0]; d2 = cKDTree(q).query(P)[0]
    return R, t, float((d1.mean() + d2.mean()) / 2 * 1000)
rep = []
for ep in sorted(CW.glob('episode_*')):
    e = int(ep.name[-6:]); names = [str(n) for n in np.load(B14 / ep.name / 'object_poses.npz', allow_pickle=False)['names']]
    for f in sorted(ep.glob('*_mesh_metric.obj')):
        n = f.name[:-len('_mesh_metric.obj')]
        if n == 'wooden_spoon': continue
        ov, of = load(OURS / ep.name / f'{n}.obj'); cv, cf = load(f)
        P = M._sample_mesh_surface(ov, of, 6000, 7); Q = M._sample_mesh_surface(cv, cf, 6000, 7)
        Up, Uq = pca(P), pca(Q); sols = []
        for sg in [(a, b, c) for a in (1, -1) for b in (1, -1) for c in (1, -1)]:
            R0 = Up @ np.diag(sg) @ Uq.T
            if np.linalg.det(R0) < 0: continue
            for extra in [np.eye(3)] + [Rot.from_euler('xyz', [a, b, c], degrees=True).as_matrix() for a, b, c in ((20, 0, 0), (0, 20, 0), (0, 0, 20))]:
                R0e = R0 @ extra if False else extra @ R0
                t0 = P.mean(0) - R0e @ Q.mean(0); sols.append(icp(P, Q, R0e, t0))
        sols.sort(key=lambda s: s[2]); R, t, cost = sols[0]
        alt = next((s for s in sols[1:] if np.degrees(np.linalg.norm(Rot.from_matrix(s[0] @ R.T).as_rotvec())) > 30), None)
        nv = cv @ R.T + t
        dst = OUT / ep.name / f'{n}.obj'; dst.parent.mkdir(exist_ok=True)
        trimesh.Trimesh(nv, cf, process=False).export(dst)
        bv, bf = K.budget_mesh(str(dst), 4096, 4096); cand = M._decode_object_mesh(K.encode(vertices=np.asarray(bv, np.float64), faces=np.asarray(bf, np.float64)))
        anc, _ = M._surface_moments(ov, of); slot = names.index(n); seeds = [1_000_003 * e + 101 * slot] + list(range(1, 10))
        regs = []
        for sd in seeds:
            ca, rot = M._register_object_body(cand, (ov, of), anc, sd)
            regs.append(dict(seed=sd, rot_deg=float(np.degrees(np.linalg.norm(Rot.from_matrix(rot).as_rotvec()))), anchor_shift_mm=float(np.linalg.norm(ca - anc) * 1000)))
        r = dict(ep=e, obj=n, slot=slot, src=str(f), out=str(dst), icp_cost_mm=cost, alt_basin_cost_mm=alt[2] if alt else None,
                 alt_basin_rot_deg=float(np.degrees(np.linalg.norm(Rot.from_matrix(alt[0] @ R.T).as_rotvec()))) if alt else None,
                 R=R.tolist(), t=t.tolist(), budget_v=len(bv), budget_f=len(bf),
                 ext_ours_cm=np.round(np.sort(np.ptp(ov, 0)) * 100, 2).tolist(), ext_theirs_cm=np.round(np.sort(np.ptp(cv, 0)) * 100, 2).tolist(),
                 kit_reg=regs, kit_max_rot_deg=max(x['rot_deg'] for x in regs), kit_pass=all(x['rot_deg'] < 5 for x in regs))
        rep.append(r); print(json.dumps({k: v for k, v in r.items() if k not in ('R', 't', 'kit_reg')}), flush=True)
(OUT / 'cwalign_report.json').write_text(json.dumps(rep, indent=1))
