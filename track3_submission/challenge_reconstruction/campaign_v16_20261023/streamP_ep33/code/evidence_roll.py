"""Stream P ep33: stereo depth residual (median mm, data->model) and silhouette IoU of the spoon at the original vs roll-corrected
pose for frames F0..F1. usage: evidence_roll.py CORRECTED_POSES.npz F0 F1 OUT.json"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import depthlib as D
zc = np.load(sys.argv[1]); f0, f1, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
z = np.load(D.FR + '/object_poses.npz'); si = [str(x) for x in z['names']].index('wooden_spoon'); wfc = z['world_from_rgb']; rows = []
for f in range(f0, f1 + 1):
    To, Tc = z['world_from_object'][f, si], zc['world_from_object'][f, si]
    if np.allclose(To, Tc): continue
    p = D.obj_points('wooden_spoon', f, wfc[f]); r0, r1 = D.residual('wooden_spoon', p, To), D.residual('wooden_spoon', p, Tc)
    i0, i1 = D.silhouette_iou('wooden_spoon', f, To, wfc[f]), D.silhouette_iou('wooden_spoon', f, Tc, wfc[f])
    rows.append(dict(frame=f, resid_orig=r0, resid_corr=r1, iou_orig=i0, iou_corr=i1, npts=len(p)))
    print(f'f{f} pts {len(p):5d} resid {r0[0]:5.1f} -> {r1[0]:5.1f} mm | IoU {i0[0]:.3f} -> {i1[0]:.3f}', flush=True)
json.dump(rows, open(out, 'w'), indent=0)
