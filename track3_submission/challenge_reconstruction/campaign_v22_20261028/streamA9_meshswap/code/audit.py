"""A9 task 1 (GT-free part): per (episode, object) of v23b.
- T_cam: v6 body -> submitted (sim/source) body, from the two camera_from_object tracks (same camera, our poses only).
- ICP of the final_v6 mesh onto the submitted mesh starting at T_cam -> T_icp, mean symmetric surface distance.
- Self-twin H of the submitted mesh (best >90 deg self-ICP solution) and its residual (how half-turn symmetric the core is).
- Ball side: the clip's ball detections (stereo camera points) mapped into the submitted body frame with the component's
  own source poses; modes (Y9 clustering); for each mode the distance to the submitted mesh vs to its twin H*mesh.
  The same test for the final_v6 mesh under the v6 poses (reference convention).
Usage: audit.py OUT.json [EP ...]"""
import sys, json, hashlib, numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from a9lib import *

def md5(p): return hashlib.md5(open(p, 'rb').read()).hexdigest()
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
def load(p): return trimesh.load(p, force='mesh', process=False)
def samp(m, n=40000, seed=0): return np.asarray(trimesh.sample.sample_surface(m, n, seed=seed)[0])
def apply(T, P): return P @ T[:3, :3].T + T[:3, 3]

def icp(src, dst_tree, dst, T0, iters=60, trim=0.9):
    T = T0.copy()
    for _ in range(iters):
        P = apply(T, src); d, i = dst_tree.query(P); k = d <= np.quantile(d, trim)
        A, B = P[k], dst[i[k]]; ca, cb = A.mean(0), B.mean(0); U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        Dg = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); Rr = Vt.T @ Dg @ U.T
        dT = np.eye(4); dT[:3, :3] = Rr; dT[:3, 3] = cb - Rr @ ca; T = dT @ T
        if np.linalg.norm(dT[:3, :3] - np.eye(3)) < 1e-7 and np.linalg.norm(dT[:3, 3]) < 1e-8: break
    return T

def symdist(A, B, tA=None, tB=None):
    tA = tA or cKDTree(A); tB = tB or cKDTree(B); return 0.5 * (tB.query(A)[0].mean() + tA.query(B)[0].mean())

def self_twin(P, tree):
    c = P.mean(0); X = P - c; _, _, Vt = np.linalg.svd(X, full_matrices=False); best = []
    for ax in list(Vt) + [(Vt[0] + Vt[1]) / np.sqrt(2), (Vt[0] - Vt[1]) / np.sqrt(2), (Vt[1] + Vt[2]) / np.sqrt(2), (Vt[0] + Vt[2]) / np.sqrt(2)]:
        T0 = np.eye(4); Rr = R.from_rotvec(np.pi * ax).as_matrix(); T0[:3, :3] = Rr; T0[:3, 3] = c - Rr @ c
        T = icp(P[::4], tree, P, T0); ang = np.degrees(R.from_matrix(T[:3, :3]).magnitude())
        if ang > 90: best.append((symdist(apply(T, P[::4]), P[::4]), T, ang))
    best.sort(key=lambda b: b[0]); return best[0] if best else (np.inf, None, None)

def modes(Q, bw=0.008, minsup=6):
    out = []; rem = np.ones(len(Q), bool)
    while rem.sum() >= minsup:
        dens = np.array([(np.linalg.norm(Q[rem] - q, axis=1) < bw).sum() for q in Q[rem]]); i = np.where(rem)[0][dens.argmax()]; c = Q[i]
        for _ in range(10): c = Q[rem][np.linalg.norm(Q[rem] - c, axis=1) < bw].mean(0)
        sel = rem & (np.linalg.norm(Q - c, axis=1) < 1.5 * bw)
        if sel.sum() < minsup: break
        out.append((c, int(sel.sum()))); rem &= ~sel
    return out

def balls_body(ep, obj, Cam):
    d = json.loads((CAN / ename(ep) / f'ball_detections_{obj}.json').read_text())['frames']; P = []
    for f, dets in enumerate(d):
        if f >= len(Cam) or not np.isfinite(Cam[f]).all(): continue
        for b in dets or []: P.append((np.array(b['camera_point']) - Cam[f][:3, 3]) @ Cam[f][:3, :3])
    return np.array(P)

def side_test(P_balls, mesh_pts, H):
    """per mode: distance to mesh surface vs to the twin H(mesh). returns list + summary"""
    if len(P_balls) == 0 or H is None: return [], None
    t0 = cKDTree(mesh_pts); t1 = cKDTree(apply(H, mesh_pts)); rows = []
    for c, k in modes(P_balls):
        rows.append(dict(c_cm=(c * 100).round(2).tolist(), n=k, d_mesh_mm=float(t0.query(c)[0] * 1000), d_twin_mm=float(t1.query(c)[0] * 1000)))
    # per-detection robust score
    dm = t0.query(P_balls)[0]; dt = t1.query(P_balls)[0]; near = np.minimum(dm, dt) < 0.015
    summ = dict(n_det=int(len(P_balls)), n_near=int(near.sum()), med_d_mesh_mm=float(np.median(dm[near]) * 1000) if near.any() else None,
                med_d_twin_mm=float(np.median(dt[near]) * 1000) if near.any() else None,
                frac_mesh_closer=float(np.mean(dm[near] + 5e-4 < dt[near])) if near.any() else None,
                frac_twin_closer=float(np.mean(dt[near] + 5e-4 < dm[near])) if near.any() else None)
    return rows, summ

def main():
    out_path = sys.argv[1]; eps = [int(e) for e in sys.argv[2:]] or [int(e) for e in SEL]; res = []
    for ep in eps:
        names_s, zs, srcdir = src_poses(ep); n6, z6 = v6_poses(ep)
        Cs_all = inv(zs['world_from_rgb'])[:, None] @ zs['world_from_object']
        for i, obj in enumerate(names_s):
            ms_path = comp_component(ep) / 'meshes' / ename(ep) / f'{obj}.obj'; m6_path = V6 / ename(ep) / 'meshes' / f'{obj}.obj'
            Ms, M6 = load(ms_path), load(m6_path); Ps, P6 = samp(Ms), samp(M6); ts = cKDTree(Ps)
            Cs = Cs_all[:, i]; C6 = z6['camera_from_object'][:, n6.index(obj)]; n = min(len(Cs), len(C6))
            ok = np.isfinite(Cs[:n]).all((1, 2)) & np.isfinite(C6[:n]).all((1, 2))
            D = inv(Cs[:n][ok]) @ C6[:n][ok]; Tc = np.eye(4); Tc[:3, :3] = rot_mean(D[:, :3, :3]); Tc[:3, 3] = np.median(D[:, :3, 3], 0)
            Ti = icp(P6[::4], ts, Ps, Tc)
            r = dict(ep=ep, obj=obj, tag=SEL[str(ep)], sub_md5=md5(ms_path)[:8], sub_sha256=sha(ms_path), v6_md5=md5(m6_path)[:8], v6_sha256=sha(m6_path),
                     same_file=md5(ms_path) == md5(m6_path), src_asset_md5=md5(srcdir / 'assets' / f'{obj}.obj')[:8],
                     Tcam_rot_spread_p50_deg=float(np.median([angdeg(d, Tc[:3, :3]) for d in D[:, :3, :3]])),
                     Tcam=Tc.tolist(), Ticp=Ti.tolist(), icp_vs_cam_deg=angdeg(Ti[:3, :3], Tc[:3, :3]), icp_vs_cam_cm=float(np.linalg.norm(Ti[:3, 3] - Tc[:3, 3]) * 100),
                     symdist_cam_cm=float(symdist(apply(Tc, P6), Ps) * 100), symdist_icp_cm=float(symdist(apply(Ti, P6), Ps) * 100))
            sd, H, ang = self_twin(Ps, ts); r.update(selftwin_symdist_cm=float(sd * 100), selftwin_deg=ang)
            if H is not None:
                r['twin_of_v6_symdist_cm'] = float(symdist(apply(H @ Ti, P6), Ps) * 100)
                Tt = icp(P6[::4], ts, Ps, H @ Ti); r['Ttwin_icp'] = Tt.tolist(); r['twin_icp_symdist_cm'] = float(symdist(apply(Tt, P6), Ps) * 100)
                r['twin_icp_vs_icp_deg'] = angdeg(Tt[:3, :3], Ti[:3, :3])
                r['selftwin_axis_body'] = (lambda rv: (rv / np.linalg.norm(rv)).round(3).tolist())(R.from_matrix(H[:3, :3]).as_rotvec())
            Bs = balls_body(ep, obj, Cs); rows, summ = side_test(Bs, Ps, H); r.update(balls_sub_modes=rows, balls_sub=summ)
            # the v6 mesh under the v6 poses, with its own self twin
            t6 = cKDTree(P6); sd6, H6, _ = self_twin(P6, t6); B6 = balls_body(ep, obj, C6)
            rows6, summ6 = side_test(B6, P6, H6); r.update(v6_selftwin_symdist_cm=float(sd6 * 100), balls_v6_modes=rows6, balls_v6=summ6)
            res.append(r)
            print(ep, obj, r['sub_md5'], r['v6_md5'], 'same' if r['same_file'] else '', f"Tcam spread {r['Tcam_rot_spread_p50_deg']:.1f} icp-cam {r['icp_vs_cam_deg']:.1f}deg/{r['icp_vs_cam_cm']:.2f}cm sd cam/icp {r['symdist_cam_cm']:.3f}/{r['symdist_icp_cm']:.3f} selftwin {r['selftwin_symdist_cm']:.3f} twin_icp {r.get('twin_icp_symdist_cm', float('nan')):.3f} ({r.get('twin_icp_vs_icp_deg', float('nan')):.0f}deg)",
                  '| sub', summ, '| v6', summ6, flush=True)
            Path(out_path).write_text(json.dumps(res, indent=1, default=float) + '\n')

if __name__ == '__main__':
    main()
