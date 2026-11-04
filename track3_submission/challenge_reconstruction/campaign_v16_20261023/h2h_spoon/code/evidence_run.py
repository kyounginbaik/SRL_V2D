"""Per-frame evidence for OURS vs NEW tracks of every object of one episode. usage: evidence_run.py EP OUT.json [STEP] [F0] [F1]"""
import sys, json, time, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import h2hlib as H, meshreg
from scipy.spatial.transform import Rotation as R
ep, out = int(sys.argv[1]), sys.argv[2]; step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
e = H.Ep(ep); f0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0; f1 = int(sys.argv[5]) if len(sys.argv) > 5 else e.N - 1
G = {}; reg = {}
for o in e.names:
    res = meshreg.register(e.mesh[('ours', o)], e.mesh[('new', o)]); G[o] = res[0][0]
    reg[o] = dict(chamfer_mm=res[0][1] * 1000, G_rotvec_deg=R.from_matrix(G[o][:3, :3]).as_rotvec(degrees=True).tolist(), G_t_cm=(G[o][:3, 3] * 100).tolist(),
                  second_chamfer_mm=res[1][1] * 1000)
pts = {o: e.samp[('ours', o)][::20] for o in e.names}
axis = {o: np.linalg.eigh(np.cov((pts[o] - pts[o].mean(0)).T))[1][:, 2] for o in e.names}
rows = []; t0 = time.time()
for f in range(f0, f1 + 1, step):
    dep = e.depth(f); hand = e.hand(f); masks = {o: e.mask(o, f) for o in e.names}; row = dict(frame=f)
    for o in e.names:
        oth = [masks[p] for p in e.names if p != o]
        Co = e.to_official('ours', e.cam_T('ours', o, f)); Cn = e.to_official('new', e.cam_T('new', o, f)) @ G[o]   # NEW expressed in OUR body frame for the pose diff only
        dcm, ddeg = H.pose_diff(Co, Cn, pts[o])
        ao, an = Co[:3, :3] @ axis[o], Cn[:3, :3] @ axis[o]
        # along-handle decomposition of the body-origin offset (in our body axes)
        off_body = Co[:3, :3].T @ (Cn[:3, 3] - Co[:3, 3]) * 100
        row[o] = dict(diff_cm=dcm, diff_deg=ddeg, axis_deg=float(np.degrees(np.arccos(np.clip(abs(ao @ an), 0, 1)))), off_body_cm=off_body.tolist(),
                      ours=e.evidence('ours', o, f, dep, masks[o], hand, oth), new=e.evidence('new', o, f, dep, masks[o], hand, oth))
    rows.append(row)
    if len(rows) % 10 == 1:
        s = ' | '.join(f"{o[:5]} d{row[o]['diff_cm']:.1f}/{row[o]['diff_deg']:.0f} dm {row[o]['ours'].get('dm', np.nan):.1f}/{row[o]['new'].get('dm', np.nan):.1f} fs {row[o]['ours'].get('fs', np.nan):.2f}/{row[o]['new'].get('fs', np.nan):.2f} iou {row[o]['ours']['iou']:.2f}/{row[o]['new']['iou']:.2f}" for o in e.names)
        print(f'f{f:3d} {s}  ({time.time() - t0:.0f}s)', flush=True)
json.dump(dict(ep=ep, names=e.names, registration=reg, depth_dir=e.depth_dir, mask_dir=e.mask_dir, hand_dirs=e.hand_dirs, rows=rows), open(out, 'w'))
