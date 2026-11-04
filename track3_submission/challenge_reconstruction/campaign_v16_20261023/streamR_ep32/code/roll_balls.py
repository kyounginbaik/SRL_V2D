"""Stream R ep32 (phase 5 task): per-frame mocap-ball reprojection residual (px, full-res evidence image, own camera model) of
  ours            : our reference spoon pose (final_recon), Kout camera
  ours_twin       : ours * Rx(180)
  new             : the coworker's NEW track (h2h_spoon/inputs_new), official cam-a model
  ours_newroll    : ours * Rx(phi_new(t)), phi_new = NEW's roll about our long body axis, measured in our camera frame
                    (NEW camera pose mapped with the h2h best-fit rotation official-a -> Kout, NEW body -> ours via meshreg G)
using the h2h canonical ball geometry (markers/canon.json) and the h2h ball detections (markers/balls_32_wooden_spoon.json).
Per detection: distance to the nearest projected canonical ball. Writes per-frame phi_new and residuals.
usage: roll_balls.py OUT_JSON"""
import sys, json, numpy as np
H2 = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon'; sys.path.insert(0, H2 + '/code')
import h2hlib as H, cammodel as CM
from scipy.spatial.transform import Rotation as R
e = H.Ep(32, nsamp=10); o = 'wooden_spoon'
cn = json.load(open(H2 + '/markers/canon.json')); canon_new = np.array(cn['canonical_cm_newframe']) / 100; G = np.array(cn['G']['32'])
canon_ours = (canon_new - G[:3, 3]) @ G[:3, :3]
balls = {r['frame']: np.array([d['px'] for d in r['dets']]) for r in json.load(open(H2 + '/markers/balls_32_wooden_spoon.json'))['rows'] if r['dets']}
Rx = lambda a: R.from_euler('x', a, degrees=True).as_matrix()
rows = []
for f in range(e.N):
    Co = e.cam_T('ours', o, f); Cn = e.R_n2o @ e.cam_T('new', o, f) @ G
    D = Co[:3, :3].T @ Cn[:3, :3]; rv = R.from_matrix(D).as_rotvec(); phi = float(np.degrees(rv[0]))   # signed roll about our body x
    off = float(np.degrees(np.linalg.norm(R.from_matrix(D @ Rx(-phi).T if False else Rx(phi).T @ D).as_rotvec())))
    r = dict(frame=f, phi_new_deg=phi, rel_deg=float(np.degrees(np.linalg.norm(rv))), off_axis_deg=off)
    if f in balls:
        uv = balls[f]
        def score(Cam, model, cb):
            X = cb @ Cam[:3, :3].T + Cam[:3, 3]; p = CM.project_full(e.cm, model, X)[0]
            return float(np.median(np.linalg.norm(uv[:, None] - p[None], axis=2).min(1)))
        Cnn = e.cam_T('new', o, f)
        r.update(n=int(len(uv)), ours=score(Co, 'ours', canon_ours), ours_twin=score(np.block([[Co[:3, :3] @ Rx(180), Co[:3, 3:]], [np.zeros((1, 3)), np.ones((1, 1))]]), 'ours', canon_ours),
                 new=score(Cnn, 'new', canon_new), ours_newroll=score(np.block([[Co[:3, :3] @ Rx(phi), Co[:3, 3:]], [np.zeros((1, 3)), np.ones((1, 1))]]), 'ours', canon_ours))
    rows.append(r)
json.dump(dict(G=G.tolist(), canonical_ours_cm=(canon_ours * 100).tolist(), rows=rows), open(sys.argv[1], 'w'), indent=1)
print(' frames  | phi_new | off | n  | ours  twin  new  ours_newroll (median px)')
for s in range(80, 190, 5):
    g = [r for r in rows if s <= r['frame'] < s + 5]; gb = [r for r in g if 'ours' in r]
    m = lambda k: np.median([r[k] for r in gb]) if gb else float('nan')
    print(f'{s:3d}-{s+4:3d} | {np.mean([r["phi_new_deg"] for r in g]):7.1f} | {np.mean([r["off_axis_deg"] for r in g]):4.1f} | {sum(r["n"] for r in gb):2d} | {m("ours"):5.1f} {m("ours_twin"):5.1f} {m("new"):5.1f} {m("ours_newroll"):5.1f}')
