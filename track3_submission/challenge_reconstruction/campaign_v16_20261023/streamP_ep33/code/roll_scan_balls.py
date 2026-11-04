"""Stream P ep33: 1-dof roll scan against the mocap balls (independent of depth/masks). For each ball detection frame of the
h2h_spoon marker set (markers/fit_33_wooden_spoon.json: frames, uv), OUR spoon pose is rolled by phi about its body x (long) axis
and the canonical cross-episode ball geometry (markers/canon.json, mapped into our mesh frame with the h2h mesh registration G)
is reprojected with OUR camera model (Kout pinhole, h2h cammodel); residual = distance of the detection to the nearest ball.
Also scores the NEW track (own camera model) for reference. usage: roll_scan_balls.py F0 F1 OUT.json"""
import sys, json, numpy as np
H2H = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon'; sys.path.insert(0, H2H + '/code')
import h2hlib as H, cammodel as CM, meshreg
from scipy.spatial.transform import Rotation as R
f0, f1, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
e = H.Ep(33, nsamp=3000); G = meshreg.register(e.mesh[('ours', 'wooden_spoon')], e.mesh[('new', 'wooden_spoon')])[0][0]
canon = np.array(json.load(open(H2H + '/markers/canon.json'))['canonical_cm_newframe']) / 100; cb_o = (canon - G[:3, 3]) @ G[:3, :3]
z = json.load(open(H2H + '/markers/fit_33_wooden_spoon.json')); F = np.array(z['frames']); UV = np.array(z['uv'])
phis = np.arange(-180, 180, 5); rows = []
for i in np.flatnonzero((F >= f0) & (F <= f1)):
    f = int(F[i]); C = e.cam_T('ours', 'wooden_spoon', f); res = []
    for ph in phis:
        Rm = C[:3, :3] @ R.from_euler('x', ph, degrees=True).as_matrix(); X = cb_o @ Rm.T + C[:3, 3]
        uv = np.stack([CM.project_full(e.cm, e.model_of('ours'), X[k:k + 1])[0][0] for k in range(3)]); res.append(float(np.linalg.norm(uv - UV[i], axis=1).min()))
    Cn = e.cam_T('new', 'wooden_spoon', f); Xn = canon @ Cn[:3, :3].T + Cn[:3, 3]
    uvn = np.stack([CM.project_full(e.cm, e.model_of('new'), Xn[k:k + 1])[0][0] for k in range(3)]); rn = float(np.linalg.norm(uvn - UV[i], axis=1).min())
    rows.append(dict(frame=f, uv=UV[i].tolist(), res_by_phi=res, ours=res[list(phis).index(0)], twin=res[list(phis).index(-180)], new=rn))
json.dump(dict(phis=phis.tolist(), rows=rows), open(out, 'w'))
fr = np.array([r['frame'] for r in rows]); M = np.array([r['res_by_phi'] for r in rows])
for g in range(f0, f1 + 1, 3):
    m = (fr >= g) & (fr < g + 3)
    if not m.any(): continue
    s = np.median(M[m], 0); b = phis[np.argmin(s)]
    print(f"f{g:3d}-{g+2:3d} n={m.sum():2d} best phi {b:5d} ({s.min():5.1f}px) | phi0 {s[list(phis).index(0)]:6.1f} | twin {s[list(phis).index(-180)]:6.1f} | -110 {s[list(phis).index(-110)]:6.1f} | new {np.median([r['new'] for r, k in zip(rows, m) if k]):6.1f}", flush=True)
