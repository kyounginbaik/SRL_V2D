"""G9 rule (GT-free; spoon only): where our spoon track and the coworker's compose27 track disagree by > 90 deg, adopt the
compose27 pose for that contiguous segment ONLY if our own 2D evidence prefers it: on the segment's frames with usable ball
detections (min(px) < 40), the compose pose's median ball reprojection is < 0.6 x ours over >= 4 frames, and its median
visible-mask IoU is not lower by more than 0.02. The segment is grown to where the disagreement drops below 45 deg and
blended (slerp / lerp) over 3 frames each side. Compose poses enter our camera through the kit registration of OUR spoon
mesh onto the compose spoon mesh plus a constant camera-frame correction fitted on agreeing frames (< 20 deg).
Our meshes are untouched; only object poses change. Writes a NEW bundle (copy of final_v6).
usage: g9_spoon_adopt.py OUT_BUNDLE [EP ...]"""
import sys, json, shutil, numpy as np, trimesh
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code')
from z9_anchor import frame_inputs, load_k, ball_px, ball_model, evidence
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
from scipy.spatial.transform import Rotation as Rot, Slerp
M = K.load_metric('MP-SR')
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026'); V6 = RR / 'selected20_final_v6'
TD = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v23_20261028/target_compose27')
import os
MODE = os.environ.get('G9_MODE', 'pose')   # 'pose' = compose pose; 'rot' = compose rotation about OUR mesh centroid, our centroid position kept
OUT = Path(sys.argv[1]); EPS = [int(e) for e in sys.argv[2:]] or [3, 5, 6, 15, 16, 32, 33, 35, 36, 37]
if not OUT.exists(): shutil.copytree(V6, OUT)
def L(p):
    m = trimesh.load(p, force='mesh'); return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
def rmean(Rs):
    q = Rot.from_matrix(Rs).as_quat(); q *= np.sign(q @ q[0])[:, None]; return Rot.from_quat(np.median(q, 0)).as_matrix()
report = {}
for ep in EPS:
    root = RR / f'canonical/evaluation/episode_{ep:06d}'; fol = V6 / f'episode_{ep:06d}'
    d = dict(np.load(fol / 'object_poses.npz')); names = [str(n) for n in d['names']]; s = names.index('wooden_spoon')
    cam = d['camera_from_object'][:, s].copy(); nf = len(cam)
    cand = L(fol / 'meshes/wooden_spoon.obj'); ref = L(TD / f'episode_{ep:06d}/wooden_spoon_mesh_metric.obj'); anc, _ = M._surface_moments(*ref)
    ca, Rb = M._register_object_body(cand, ref, anc, 1_000_003 * ep + 101 * s)
    Mo2 = np.eye(4); Mo2[:3, :3] = Rb; Mo2[:3, 3] = anc - Rb @ ca
    Tc = np.load(TD / f'episode_{ep:06d}/wooden_spoon_trajectory.npz')['T_world_obj']; cmo = np.load(TD / f'episode_{ep:06d}/camera_motion.npz')
    n = min(nf, len(Tc))
    imp = cmo['T_a_from_rectL'] @ np.linalg.inv(cmo['cam0_T_cam'][:n]) @ Tc[:n] @ Mo2
    ang = np.degrees(Rot.from_matrix(np.einsum('fji,fjk->fik', cam[:n, :3, :3], imp[:, :3, :3])).magnitude())
    agree = np.where(ang < 20)[0]
    E = np.eye(4)
    if len(agree) >= 10:
        Ec = cam[agree] @ np.linalg.inv(imp[agree]); E[:3, :3] = rmean(Ec[:, :3, :3]); E[:3, 3] = np.median(Ec[:, :3, 3], 0)
    imp = E @ imp
    if MODE == 'rot':
        cen = M._surface_moments(*cand)[0]
        for f in range(n):
            pc = cam[f, :3, :3] @ cen + cam[f, :3, 3]; imp[f, :3, 3] = pc - imp[f, :3, :3] @ cen
    ang = np.degrees(Rot.from_matrix(np.einsum('fji,fjk->fik', cam[:n, :3, :3], imp[:, :3, :3])).magnitude())
    mesh = trimesh.load(fol / 'meshes/wooden_spoon.obj', force='mesh', process=False)
    pts, fc = trimesh.sample.sample_surface(mesh, 20000, seed=260126); nrm = np.asarray(mesh.face_normals)[fc]
    k, intr = load_k(root, 507, 380); kf = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    model, dets = ball_model(root, 'wooden_spoon')
    segs = []; t = 0
    while t < n:
        if ang[t] > 90:
            a = t
            while t < n and ang[t] > 90: t += 1
            segs.append([a, t - 1])
        else: t += 1
    # merge segments separated by < 4 frames
    mg = []
    for sg in segs:
        if mg and sg[0] - mg[-1][1] <= 4: mg[-1][1] = sg[1]
        else: mg.append(sg)
    new = cam.copy(); fired = []
    for a, b in mg:
        bo, bc, io, ic = [], [], [], []
        for f in range(a, b + 1):
            nb = dets[f] if f < len(dets) else []
            if nb:
                po, pc = ball_px(model, nb, cam[f], kf), ball_px(model, nb, imp[f], kf)
                if min(po, pc) < 40: bo.append(po); bc.append(pc)
            fi = frame_inputs(root, 'wooden_spoon', names, f, 507, 380)
            if fi is not None:
                m, ex, dp, _ = fi
                io.append(evidence(mesh, pts, nrm, cam[f], k, m, ex, dp)[1]['visible_iou']); ic.append(evidence(mesh, pts, nrm, imp[f], k, m, ex, dp)[1]['visible_iou'])
        ok = len(bo) >= 4 and np.median(bc) < 0.6 * np.median(bo) and (not io or np.median(ic) >= np.median(io) - 0.02)
        rec = dict(seg=[a, b], ball_frames=len(bo), ours_ball=float(np.median(bo)) if bo else None, comp_ball=float(np.median(bc)) if bc else None,
                   ours_iou=float(np.median(io)) if io else None, comp_iou=float(np.median(ic)) if ic else None, adopt=bool(ok))
        if ok:
            a2, b2 = a, b
            while a2 > 0 and ang[a2 - 1] > 45: a2 -= 1
            while b2 < n - 1 and ang[b2 + 1] > 45: b2 += 1
            new[a2:b2 + 1] = imp[a2:b2 + 1]; rec['applied'] = [a2, b2]; fired.append([a2, b2])
            for lo, hi, inside in [(max(a2 - 3, 0), a2, a2), (b2, min(b2 + 3, n - 1), b2)]:
                for f in range(lo, hi + 1):
                    if f in (a2, b2) and lo != hi: continue
                    w = 1 - abs(f - inside) / 4.0
                    sl = Slerp([0, 1], Rot.from_matrix(np.stack([cam[f, :3, :3], imp[f, :3, :3]])))
                    new[f, :3, :3] = sl([w]).as_matrix()[0]; new[f, :3, 3] = (1 - w) * cam[f, :3, 3] + w * imp[f, :3, 3]
        report.setdefault(ep, dict(mode=MODE, E_rot_deg=float(np.degrees(Rot.from_matrix(E[:3, :3]).magnitude())), E_t_mm=float(np.linalg.norm(E[:3, 3]) * 1000),
                                   agree_frames=int(len(agree)), segments=[]))['segments'].append(rec)
        print(ep, rec, flush=True)
    if fired:
        d['camera_from_object'] = d['camera_from_object'].copy(); d['camera_from_object'][:, s] = new
        d['world_from_object'] = d['world_from_object'].copy(); d['world_from_object'][:, s] = d['world_from_rgb'] @ new
        np.savez(OUT / f'episode_{ep:06d}' / 'object_poses.npz', **d)
        (OUT / f'episode_{ep:06d}' / 'g9_spoon_adopt.json').write_text(json.dumps(report[ep], indent=1))
    report.setdefault(ep, {})['fired'] = fired
    print(ep, 'fired', fired, flush=True)
json.dump(report, open(OUT / f'g9_spoon_adopt_report_{"_".join(map(str, EPS))}.json', 'w'), indent=1)
