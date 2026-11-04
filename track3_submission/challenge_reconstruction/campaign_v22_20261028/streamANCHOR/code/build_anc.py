"""ANCHOR target builder (GT-free): copy streamALL20K/targets/tgt_v32/episode_EP to streamALL20K/targets/LABEL/episode_EP and correct the
frame-0 scene-alignment anchors over the static start window [0, t_s] with a smoothstep ramp back to the tracked pose by t_e = t_s + RAMP:
  rot : rotation anchor rotated by its physical settle on OUR local stereo table plane (audit_epEP.json: median camera-frame settle
        rotvec over the audited static frames, mapped to world with world_from_rgb), about its COM, times SCALE;
  h   : translation anchor (slot 0) moved along the table normal so its lowest point touches the plane (median lowest_mm), times SCALE.
Every other object/frame unchanged. RAMP='all' = constant correction over the whole episode (never-manipulated anchors, e.g. the rack).
usage: build_anc.py EP LABEL rot|h|both [SCALE=1] [RAMP=15|all]"""
import sys, json, shutil, hashlib, numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
EP, LAB, MODE = int(sys.argv[1]), sys.argv[2], sys.argv[3]; SC = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
RAMP = sys.argv[5] if len(sys.argv) > 5 else '15'; ALL = RAMP == 'all'; RAMP = 15 if ALL else int(RAMP); NM = f'episode_{EP:06d}'
C22 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'); A = C22 / 'streamANCHOR'
SRC = C22 / 'streamALL20K/targets/tgt_v32' / NM; DST = C22 / 'streamALL20K/targets' / LAB / NM
REC = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
au = json.loads((A / f'out/audit_ep{EP}.json').read_text())
assert not DST.exists(), DST; shutil.copytree(SRC, DST)
Wrgb = np.load(REC / 'selected20_final_v14_g17' / NM / 'object_poses.npz', allow_pickle=False)['world_from_rgb']
sm = lambda u: np.clip(u, 0, 1) ** 3 * (10 - 15 * np.clip(u, 0, 1) + 6 * np.clip(u, 0, 1) ** 2)
def ramp(n, ts):
    if ALL: return np.ones(n)
    te = ts + RAMP; return np.array([1.0 if t <= ts else (1 - sm((t - ts) / (te - ts)) if t < te else 0.0) for t in range(n)])
rec = dict(stream='ANCHOR', label=LAB, ep=EP, mode=MODE, scale=SC, ramp='all' if ALL else RAMP, source=str(SRC), ground_truth_used=False, changes={})
def load(o): return np.load(DST / f'{o}_trajectory.npz', allow_pickle=False)['T_world_obj'].copy()
def save(o, T):
    (DST / f'{o}_trajectory.npz').unlink(); np.savez_compressed(DST / f'{o}_trajectory.npz', T_world_obj=T)
def com_of(o):
    m = trimesh.load(DST / f'{o}_mesh_metric.obj', force='mesh', process=False)
    return np.asarray(m.center_mass) if m.is_watertight else (m.triangles_center * m.area_faces[:, None]).sum(0) / m.area
fr = lambda o: [(int(t), x) for t, x in au['objects'][o]['frames'].items() if x]
if MODE in ('rot', 'both'):
    o = au['rotation_anchor']; assert o in au['slots'], 'no single rotation anchor (all-symmetric episode)'
    T = load(o); ts = au['objects'][o]['static_until'] - 1; w = ramp(len(T), ts); com = com_of(o)
    rvw = np.median([Wrgb[t][:3, :3] @ np.radians(x['settle_rotvec_cam_deg']) for t, x in fr(o)], axis=0) * SC
    for t in np.flatnonzero(w > 0):
        R = Rot.from_rotvec(rvw * w[t]).as_matrix(); c = T[t][:3, :3] @ com + T[t][:3, 3]
        T[t][:3, :3] = R @ T[t][:3, :3]; T[t][:3, 3] = R @ (T[t][:3, 3] - c) + c
    save(o, T); rec['changes'][o] = dict(kind='rot', world_rotvec_deg=np.degrees(rvw).round(3).tolist(), deg=float(np.degrees(np.linalg.norm(rvw))), t_s=ts,
                                         frames=int((w > 0).sum()))
if MODE == 'camall':
    # constant CAMERA-frame de-tilt of the rotation anchor on EVERY frame (bias stable over start/end rests, audit + audit_late):
    # R_w(t) = Rwc(t) R_c Rwc(t)^T about the COM; R_c = median settle rotvec over on-table frames (|lowest| < 25 mm) of both audits.
    import glob
    o = au['rotation_anchor']; T = load(o); com = com_of(o); vs = []
    for f in [A / f'out/audit_ep{EP}.json'] + sorted(glob.glob(str(A / f'out/audit_ep{EP}_f*.json'))):
        for t, x in json.loads(Path(f).read_text())['objects'][o]['frames'].items():
            if x and abs(x['lowest_mm']) < 25: vs.append(np.radians(x['settle_rotvec_cam_deg']))
    rc = np.median(vs, axis=0) * SC
    for t in range(len(T)):
        Rw = Wrgb[t][:3, :3]; R = Rw @ Rot.from_rotvec(rc).as_matrix() @ Rw.T; c = T[t][:3, :3] @ com + T[t][:3, 3]
        T[t][:3, :3] = R @ T[t][:3, :3]; T[t][:3, 3] = R @ (T[t][:3, 3] - c) + c
    save(o, T); rec['changes'][o] = dict(kind='camall', cam_rotvec_deg=np.degrees(rc).round(3).tolist(), deg=float(np.degrees(np.linalg.norm(rc))), n_frames_est=len(vs), frames=len(T))
if MODE in ('h', 'both'):
    o = au['translation_anchor']; T = load(o); ts = au['objects'][o]['static_until'] - 1; w = ramp(len(T), ts)
    # normal in world from the camera-frame plane normal: settle uses n up; lowest_mm along n -> move by -lowest along n
    zc = json.loads((A / f'out/audit_ep{EP}.json').read_text())
    low = np.median([x['lowest_mm'] for _, x in fr(o)]) / 1000 * SC
    nw = np.median([Wrgb[t][:3, :3] @ np.asarray(x.get('n_cam', [0, 0, 0])) for t, x in fr(o)], axis=0)
    assert np.linalg.norm(nw) > 0.5, 'audit lacks n_cam'
    nw /= np.linalg.norm(nw)
    for t in np.flatnonzero(w > 0): T[t][:3, 3] -= nw * low * w[t]
    save(o, T); rec['changes'][o + '_h'] = dict(kind='height', move_mm=float(-1000 * low), normal_world=nw.round(4).tolist(), t_s=ts, frames=int((w > 0).sum()))
rec['sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DST.glob('*_trajectory.npz'))}
(DST / 'ANCHOR_CHANGES.json').write_text(json.dumps(rec, indent=1)); print(json.dumps(rec['changes']))
