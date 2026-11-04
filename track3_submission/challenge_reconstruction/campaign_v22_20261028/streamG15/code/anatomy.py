"""Scoring-only error anatomy of one export on one public clip. Usage: anatomy.py EP TAG -> out/anat_<TAG>_<EP>.json"""
import sys, json, numpy as np
from scipy.spatial.transform import Rotation as Rot
import kitx
ep, tag = int(sys.argv[1]), sys.argv[2]
X = kitx.load_export(ep, tag)
o = kitx.score(ep, {n: X['W'][:, i] for i, n in enumerate(X['names'])}, X['meshes'], cam=X['cam'])
F = o['F']; Rc, cc = o['camR'], o['camc']; M = kitx.M
def rm(q): return M._rotation_matrix(q)
def stats(v):  # v [F,3]
    m = v.mean(0); return dict(mean_norm=float(np.linalg.norm(v, axis=1).mean()), const=m.tolist(), const_norm=float(np.linalg.norm(m)),
                               resid=float(np.linalg.norm(v - m, axis=1).mean()))
res = dict(ep=ep, tag=tag, F=F, objs={})
gtspeed = {}
for s, n in enumerate(o['objs']):
    Ap, Rp = o['A'][:, s, :3], o['R'][:, s, :3]; Ra, Rr = rm(o['A'][:, s, 3:]), rm(o['R'][:, s, 3:])
    e = (Ap - Rp) * 100
    eb = np.einsum('tji,tj->ti', Rr, e)
    ec = np.einsum('tji,tj->ti', Rc, e)
    u = np.einsum('tji,tj->ti', Rc, Ap - cc); dist = np.linalg.norm(u, axis=1); u /= dist[:, None]
    along = (ec * u).sum(1); lat = np.linalg.norm(ec - along[:, None] * u, axis=1)
    ax, od = o['axes'][s], o['orders'][s]
    if od == 360:
        aa, ar = Ra @ ax, Rr @ ax
        cr = np.cross(ar, aa); sn = np.linalg.norm(cr, axis=1); ang = np.arctan2(sn, (aa * ar).sum(1))
        ww = cr / np.maximum(sn, 1e-12)[:, None] * ang[:, None]
        wb = np.einsum('tji,tj->ti', Rr, ww)
    else:
        if od == 2:
            flip = Rot.from_rotvec(np.pi * ax).as_matrix()
            d1 = Rot.from_matrix(np.einsum('tij,tkj->tik', Ra, Rr)).magnitude()
            Rr2 = Rr @ flip
            d2 = Rot.from_matrix(np.einsum('tij,tkj->tik', Ra, Rr2)).magnitude()
            Rr = np.where((d2 < d1)[:, None, None], Rr2, Rr)
        ww = Rot.from_matrix(np.einsum('tij,tkj->tik', Ra, Rr)).as_rotvec()
        wb = Rot.from_matrix(np.einsum('tji,tjk->tik', Rr, Ra)).as_rotvec()
    wc = np.einsum('tji,tj->ti', Rc, ww)
    D = np.degrees
    sp = np.r_[0, np.linalg.norm(np.diff(Rp, axis=0), axis=1)] * 20 * 100  # cm/s GT
    gtspeed[n] = sp
    rb = np.linalg.norm(eb - eb.mean(0), axis=1)
    static = sp < 2.0
    res['objs'][n] = dict(
        pos=float(o['pos'][:, s].mean()), ori=float(o['ori'][:, s].mean()), f0pos=float(o['pos'][0, s]), f0ori=float(o['ori'][0, s]),
        dist_m=float(dist.mean() / 100 * 100) / 100 if False else float(dist.mean()),
        body=stats(eb), cam=stats(ec), world=stats(e), along_mean=float(along.mean()), along_absmean=float(np.abs(along).mean()), lat_mean=float(lat.mean()),
        rot_body={k: (D(v) if isinstance(v, float) else D(np.array(v)).tolist()) for k, v in stats(wb).items()},
        rot_cam={k: (D(v) if isinstance(v, float) else D(np.array(v)).tolist()) for k, v in stats(wc).items()},
        rot_world={k: (D(v) if isinstance(v, float) else D(np.array(v)).tolist()) for k, v in stats(ww).items()},
        static_frac=float(static.mean()), resid_body_static=float(rb[static].mean()) if static.any() else None,
        resid_body_moving=float(rb[~static].mean()) if (~static).any() else None,
        order=int(od), symmetric=bool(np.any(ax)))
    np.savez_compressed(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15/out/pf_{tag}_{ep}_{n}.npz', e=e, eb=eb, ec=ec, ww=D(ww), wb=D(wb), wc=D(wc), along=along, dist=dist, sp=sp, pos=o['pos'][:, s], ori=o['ori'][:, s])
# oracle all-frame scene alignment (keypoints at +-5 cm along body axes; symmetric axes only along the axis)
P, Q = [], []
for s, n in enumerate(o['objs']):
    Ra, Rr = rm(o['A'][:, s, 3:]), rm(o['R'][:, s, 3:]); ax = o['axes'][s]
    ks = [np.zeros(3)] + ([ax * .05, -ax * .05] if o['orders'][s] == 360 else [] if np.any(ax) else [np.eye(3)[i] * .05 * sg for i in range(3) for sg in (1, -1)])
    for k in ks:
        P.append(o['A'][:, s, :3] + Ra @ k); Q.append(o['R'][:, s, :3] + Rr @ k)
P, Q = np.concatenate(P), np.concatenate(Q)
cp, cq = P.mean(0), Q.mean(0); U, _, Vt = np.linalg.svd((P - cp).T @ (Q - cq)); d = np.sign(np.linalg.det(Vt.T @ U.T))
S2 = Vt.T @ np.diag([1, 1, d]) @ U.T; t2 = cq - S2 @ cp
res['oracle_align'] = dict(rot_deg=float(np.degrees(Rot.from_matrix(S2).magnitude())), trans_cm=float(np.linalg.norm(t2 + S2 @ np.zeros(3)) * 100))
for s, n in enumerate(o['objs']):
    Ap2 = o['A'][:, s, :3] @ S2.T + t2
    res['objs'][n]['pos_oracle_align'] = float(np.linalg.norm(Ap2 - o['R'][:, s, :3], axis=1).mean() * 100)
json.dump(res, open(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15/out/anat_{tag}_{ep}.json', 'w'), indent=1)
for n, r in res['objs'].items():
    print(f"{ep:3d} {n:20s} pos {r['pos']:.2f} ori {r['ori']:.1f} | body const {r['body']['const_norm']:.2f} resid {r['body']['resid']:.2f} | cam const {r['cam']['const_norm']:.2f} resid {r['cam']['resid']:.2f} along {r['along_mean']:+.2f} | world const {r['world']['const_norm']:.2f} resid {r['world']['resid']:.2f} | rot body const {r['rot_body']['const_norm']:.1f} resid {r['rot_body']['resid']:.1f} | orac {r['pos_oracle_align']:.2f}")
