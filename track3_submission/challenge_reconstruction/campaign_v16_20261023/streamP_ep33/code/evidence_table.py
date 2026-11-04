"""Stream P ep33: evidence for the object-in-table frames. For each frame: the setup-frame lowest point of the reference object,
the local stereo table height around it, and the stereo depth residual of the object at the original pose vs the pose lifted
vertically so its lowest point sits on the stereo table (setup z). usage: evidence_table.py OBJ T0 T1 OUT.json"""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import depthlib as D
obj, t0, t1, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
z = np.load(D.FR + '/object_poses.npz'); names = [str(x) for x in z['names']]; W = z['world_from_object'][:, names.index(obj)]; wfc = z['world_from_rgb']
Sinv = np.linalg.inv(D.SETUP); rows = []
for f in range(int(round(t0 * 20)), int(round(t1 * 20)) + 1):
    T = W[f]; Ts = D.SETUP @ T; low = float((D.MESH[obj].vertices @ Ts[:3, :3].T + Ts[:3, 3])[:, 2].min())
    tz, spread, npt = D.table_plane_z(f, wfc[f], T[:3, 3])
    lift = max(0.0, (0.0 if not np.isfinite(tz) else tz) - low)
    Tl = Ts.copy(); Tl[2, 3] += max(0.0, -low); Tl = Sinv @ Tl     # lift to the physics table z=0
    P = D.obj_points(obj, f, wfc[f])
    r0 = D.residual(obj, P, T); r1 = D.residual(obj, P, Tl)
    rows.append(dict(t=f / 20, low_cm=low * 100, stereo_table_z_cm=tz * 100, table_spread_cm=spread * 100, table_pts=npt, n_obj_pts=len(P),
                     resid_orig_mm=r0, resid_lift_to_z0_mm=r1, lift_cm=max(0.0, -low) * 100))
    print(f"{f/20:5.2f} low {low*100:6.2f} stereo-table {tz*100:6.2f} (iqr {spread*100:4.2f}, n {npt}) | {obj} pts {len(P):5d} resid med/trim orig {r0[0]:5.1f}/{r0[1]:5.1f} lifted {r1[0]:5.1f}/{r1[1]:5.1f} mm (lift {max(0,-low)*100:4.2f} cm)", flush=True)
json.dump(rows, open(out, 'w'), indent=0)
