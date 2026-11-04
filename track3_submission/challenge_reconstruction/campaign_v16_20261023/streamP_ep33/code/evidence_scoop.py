"""Stream P ep33: evidence for the spoon<->cup-wall interpenetration frames. For each frame: collision-part penetration depth
at the reference poses; minimal translation of the SPOON out of the cup (hypothesis S) and of the CUP away from the spoon
(hypothesis C); stereo depth residual (median / trimmed mean, mm) and silhouette IoU of the spoon and the cup at the original
vs the moved pose. usage: evidence_scoop.py T0 T1 OUT.json [STEP]"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import depthlib as D; from depen import Collider
t0, t1, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]; step = int(sys.argv[4]) if len(sys.argv) > 4 else 1
z = np.load(D.FR + '/object_poses.npz'); names = [str(x) for x in z['names']]; W = z['world_from_object']; wfc = z['world_from_rgb']
S, Si = D.SETUP, np.linalg.inv(D.SETUP); col = Collider(); rows = []
for f in range(int(round(t0 * 20)), int(round(t1 * 20)) + 1, step):
    P = {n: S @ W[f, i] for i, n in enumerate(names)}; dep0 = col.depth(P, 'blue_cup', 'wooden_spoon')
    Ts, ts = col.push_out(P, 'wooden_spoon', 'blue_cup'); Tc, tc = col.push_out(P, 'blue_cup', 'wooden_spoon')
    ps, pc = D.obj_points('wooden_spoon', f, wfc[f]), D.obj_points('blue_cup', f, wfc[f])
    r = dict(t=f / 20, pen_mm=dep0 * 1000, spoon_shift_cm=float(np.linalg.norm(ts) * 100), spoon_shift_setup_cm=(ts * 100).round(2).tolist(),
             cup_shift_cm=float(np.linalg.norm(tc) * 100), n_spoon_pts=len(ps), n_cup_pts=len(pc),
             spoon_resid_orig=D.residual('wooden_spoon', ps, W[f, 1]), spoon_resid_S=D.residual('wooden_spoon', ps, Si @ Ts),
             cup_resid_orig=D.residual('blue_cup', pc, W[f, 0]), cup_resid_C=D.residual('blue_cup', pc, Si @ Tc),
             spoon_iou_orig=D.silhouette_iou('wooden_spoon', f, W[f, 1], wfc[f]), spoon_iou_S=D.silhouette_iou('wooden_spoon', f, Si @ Ts, wfc[f]),
             cup_iou_orig=D.silhouette_iou('blue_cup', f, W[f, 0], wfc[f]), cup_iou_C=D.silhouette_iou('blue_cup', f, Si @ Tc, wfc[f]))
    rows.append(r)
    print(f"{r['t']:5.2f} pen {r['pen_mm']:5.1f} | S shift {r['spoon_shift_cm']:4.2f} cm resid {r['spoon_resid_orig'][0]:5.1f}->{r['spoon_resid_S'][0]:5.1f} iou {r['spoon_iou_orig'][0]:.3f}->{r['spoon_iou_S'][0]:.3f} "
          f"| C shift {r['cup_shift_cm']:4.2f} cm resid {r['cup_resid_orig'][0]:5.1f}->{r['cup_resid_C'][0]:5.1f} iou {r['cup_iou_orig'][0]:.3f}->{r['cup_iou_C'][0]:.3f}", flush=True)
json.dump(rows, open(out, 'w'), indent=0)
