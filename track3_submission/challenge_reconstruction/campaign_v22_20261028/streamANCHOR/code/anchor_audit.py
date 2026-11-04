"""ANCHOR audit (GT-free): for one eval episode, the kit's frame-0 scene-alignment anchors on OUR v14_g17 reconstruction
(= what the submitted f0 rows carry) and physical checks of every object resting on the table in the static start window:
  slot order = results/episode_*.parquet of the current best package; kit rule (_align_registered_scene): rotation anchor = first
  slot whose reference has no symmetry axis (non-symmetric; symmetric = beige_cup/blue_cup 360, white_pot_lid 2), else slot0 axis +
  slot0->slot1 direction; translation anchor = slot 0 f0 position.
  (a) height: lowest mesh point vs a LOCAL stereo table plane (table32.py method: RANSAC on our depth in a 6-25 cm ring around the
      object, outside all object masks), frames F;
  (b) settle: physical rest of our rigid mesh on that plane = nearest local minimum of COM height (2-DOF tilt descent about the two
      in-plane axes through the COM; steps 2/1/0.5/0.25 deg) -> settle angle + its long-axis roll component; plus the roll-only global
      min over +-60 deg (build_k32r rule);
  (c) lever: cm of mean object-point error per 1 deg anchor rotation (mean |p_t - p0_slot0| + 5 cm MPPE keypoint arm), and per 1 mm
      anchor translation = 0.1 cm.
usage: anchor_audit.py EP -> out/audit_epEP.json"""
import sys, json, numpy as np, cv2, trimesh, pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
EP = int(sys.argv[1]); NM = f'episode_{EP:06d}'
FX = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else None   # optional frames -> out/audit_epEP_fFIRST-LAST.json
REC = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
B = REC / 'selected20_final_v14_g17' / NM; C = REC / 'canonical/evaluation' / NM
TG = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamALL20K/targets/tgt_v32') / NM
PK = Path('/data1/SRL_V2D/challenge_submissions_SRL/track3_v33r16c3_ep32r10_20261102/all_20_episodes')
OUT = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamANCHOR/out')
SYM = {'beige_cup': 360, 'blue_cup': 360, 'white_pot_lid': 2}
r = pd.read_parquet(PK / f'results/{NM}.parquet', columns=['object_id', 'object_body_name']).drop_duplicates().sort_values('object_id')
slots = r.object_body_name.tolist()
z = np.load(B / 'object_poses.npz', allow_pickle=False); names = list(z['names'])
W = {o: z['world_from_object'][:, names.index(o)] for o in slots}; Cm = {o: z['camera_from_object'][:, names.index(o)] for o in slots}
K = json.load(open(C / 'intrinsics.json')); fx, fy, cx, cy, Wd, Hd = K['fx'], K['fy'], K['cx'], K['cy'], K['width'], K['height']
uu, vv = np.meshgrid(np.arange(Wd), np.arange(Hd))
def mesh(o):
    m = trimesh.load(TG / f'{o}_mesh_metric.obj', force='mesh', process=False); v = np.asarray(m.vertices, float)
    com = np.asarray(m.center_mass) if m.is_watertight else (m.triangles_center * m.area_faces[:, None]).sum(0) / m.area
    try: v = np.asarray(m.convex_hull.vertices, float)        # lowest point / rest potential only depend on the hull
    except Exception: pass
    return v, com
def rd(p): return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
def occluders(t):
    occ = np.zeros((Hd, Wd), bool)
    for d in (C / 'objects').iterdir():
        for k in '123':
            m = rd(d / 'masks' / k / f'{t:06d}.png')
            if m is not None: occ |= m > 0
    for hp in (C / 'hands').glob('*/masks/*/' + f'{t:06d}.png') if (C / 'hands').exists() else []:
        occ |= rd(hp) > 0
    return cv2.dilate(occ.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
def local_plane(t, ctr, occ):
    dv = rd(C / 'depth' / f'{t:06d}.png')
    if dv is None: return None
    dv = dv.astype(np.float64); D = np.where(dv < 65535, 65535 / np.maximum(dv, 1) - 1, np.nan)
    P = np.stack([(uu - cx) / fx * D, (vv - cy) / fy * D, D], -1)
    dist = np.linalg.norm(P - ctr, axis=-1); sel = (~occ) & np.isfinite(D) & (dist > 0.06) & (dist < 0.25); Q = P[sel]
    if len(Q) < 500: return None
    rng = np.random.default_rng(0); best = None
    for _ in range(400):
        s = Q[rng.choice(len(Q), 3, replace=False)]; n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n); inl = np.abs((Q - s[0]) @ n) < 0.004
        if best is None or inl.sum() > best[0].sum(): best = (inl, n)
    q = Q[best[0]]; qc = q.mean(0); n = np.linalg.svd(q - qc, full_matrices=False)[2][2]
    if n @ (-qc) < 0: n = -n                        # normal toward the camera = up
    return n, qc, int(best[0].sum()), float(1000 * np.std((q - qc) @ n))
def settle(vc, comc, n):
    """vc, comc: mesh vertices / COM in camera frame; n: up normal. Nearest local min of potential U = n.(com - lowest vertex)."""
    a = np.cross(n, [1, 0, 0]); a /= np.linalg.norm(a); b = np.cross(n, a)
    def U(x):
        R = Rot.from_rotvec(np.radians(x[0]) * a + np.radians(x[1]) * b).as_matrix(); w = (vc - comc) @ R.T
        return -(w @ n).min()
    x = np.zeros(2); u = U(x)
    for st in (2.0, 1.0, 0.5, 0.25):
        moved = True
        while moved:
            moved = False
            for dx in ((st, 0), (-st, 0), (0, st), (0, -st), (st, st), (st, -st), (-st, st), (-st, -st)):
                y = x + dx; uy = U(y)
                if uy < u - 1e-7 and np.hypot(*y) < 60: x, u, moved = y, uy, True
    Rs = Rot.from_rotvec(np.radians(x[0]) * a + np.radians(x[1]) * b)
    return Rs, float(np.degrees(Rs.magnitude())), float(1000 * (U(np.zeros(2)) - u))
res = dict(ep=EP, slots=slots, symmetric={o: SYM.get(o, 0) for o in slots})
fixed = [i for i, o in enumerate(slots) if o not in SYM]
res['rotation_anchor'] = slots[fixed[0]] if fixed else f'{slots[0]} axis + {slots[0]}->{slots[1]} direction'
res['translation_anchor'] = slots[0]
nf = len(W[slots[0]]); p0 = W[slots[0]][0][:3, 3]
lev = []
for o in slots:
    v, com = mesh(o); pts = (W[o][:, :3, :3] @ com) + W[o][:, :3, 3]; lev.append(np.linalg.norm(pts - p0, axis=1).mean() + 0.05)
res['lever_cm_per_deg'] = float(np.mean(lev) * 100 * np.pi / 180)
F = FX or [0, 3, 6, 10]
objs = {}
for o in slots:
    v, com = mesh(o); long = np.linalg.eigh(np.cov((v - v.mean(0)).T))[1][:, 2]
    rec = dict(frames={})
    for t in F:
        if t >= nf: continue
        Tc = Cm[o][t]; vc = v @ Tc[:3, :3].T + Tc[:3, 3]; comc = Tc[:3, :3] @ com + Tc[:3, 3]
        occ = occluders(t); lp = local_plane(t, vc.mean(0), occ)
        if lp is None: rec['frames'][t] = None; continue
        n, qc, ni, rms = lp; h = (vc - qc) @ n
        Rs, ang, dU = settle(vc, comc, n)
        lc = Tc[:3, :3] @ long; rv = Rs.as_rotvec(); roll = float(np.degrees(rv @ lc))
        rec['frames'][t] = dict(lowest_mm=float(1000 * h.min()), com_mm=float(1000 * ((comc - qc) @ n)), inliers=ni, plane_rms_mm=rms,
                                settle_deg=ang, settle_roll_deg=roll, settle_rotvec_cam_deg=np.degrees(rv).round(2).tolist(), dU_mm=dU, n_cam=n.round(5).tolist())
    fr = [x for x in rec['frames'].values() if x]
    if fr:
        rec['median_lowest_mm'] = float(np.median([x['lowest_mm'] for x in fr]))
        rec['median_settle_deg'] = float(np.median([x['settle_deg'] for x in fr]))
        rec['median_settle_roll_deg'] = float(np.median([x['settle_roll_deg'] for x in fr]))
        rec['on_table'] = abs(rec['median_lowest_mm']) < 20
    # static window length: frames until the object's COM moves > 1 cm from f0
    pc = (W[o][:, :3, :3] @ com) + W[o][:, :3, 3]; mv = np.flatnonzero(np.linalg.norm(pc - pc[0], axis=1) > 0.01)
    rec['static_until'] = int(mv[0]) if len(mv) else nf
    objs[o] = rec
res['objects'] = objs
(OUT / (f'audit_ep{EP}.json' if FX is None else f'audit_ep{EP}_f{FX[0]}-{FX[-1]}.json')).write_text(json.dumps(res, indent=1))
print(EP, res['rotation_anchor'], '| T:', res['translation_anchor'], f"| lever {res['lever_cm_per_deg']:.3f} cm/deg |",
      ' ; '.join(f"{o}: low {d.get('median_lowest_mm', float('nan')):+.1f}mm settle {d.get('median_settle_deg', float('nan')):.1f}deg "
                 f"(roll {d.get('median_settle_roll_deg', float('nan')):+.1f}) static<{d['static_until']}" for o, d in objs.items()), flush=True)
