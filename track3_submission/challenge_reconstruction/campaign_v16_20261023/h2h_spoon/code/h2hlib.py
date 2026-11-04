"""H2H spoon: loaders for OUR reference track (final_recon / remaining7 / ep36 corrected) and the NEW compose_eval track,
both expressed in the RGB camera ('a', undistorted) of each frame, plus per-frame evidence against the stereo depth_rgb and
SAM2 masks (same evidence for both tracks; each track with its own paired mesh)."""
import json, numpy as np, cv2, trimesh, os
import cammodel
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
ROOT = '/data1/SRL_V2D/challenge_reconstruction'
E = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/inputs_new'  # copy of compose_eval_20261021 (trajectory npz, mesh obj, camera_motion)
HAW = '/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs'
def fr_dir(ep):
    return f'{ROOT}/dexagent_remaining7_20261019/input_root/challenge_reconstruction/final_recon/episode_{ep:06d}' if ep in (15, 16) else f'{ROOT}/final_recon/episode_{ep:06d}'
def ours_poses_path(ep):
    if ep == 36: return f'{ROOT}/dexagent_final_recon_20261017/refinement_20261021_ep36/source_corrected/episode_000036/object_poses.npz'
    return fr_dir(ep) + '/object_poses.npz'
class Ep:
    def __init__(s, ep, nsamp=60000):
        s.ep = ep; s.fr = fr_dir(ep); z = np.load(ours_poses_path(ep)); s.names = [str(n) for n in z['names']]
        s.W = z['world_from_object']; s.wfc = z['world_from_rgb'].astype(np.float64); s.N = len(s.W)
        cfg = json.load(open(s.fr + '/config.json'))
        s.depth_dir = cfg.get('depth') or f'{ROOT}/eval_stereo_current_20261010/episode_{ep:06d}/depth_rgb'
        if not os.path.isdir(s.depth_dir): s.depth_dir = f'{ROOT}/eval_stereo_current_20261010/episode_{ep:06d}/depth_rgb'
        s.mask_dir = {o: (cfg.get('object_masks') or {}).get(o, f'{HAW}/episode_{ep:06d}/{o}/masks/1') for o in s.names}
        s.hand_dirs = [f'{HAW}/episode_{ep:06d}/{s.names[-1]}/masks/{t}' for t in ('2', '3')]
        K = json.load(open(f'{HAW}/episode_{ep:06d}/{s.names[-1]}/anycalib/undistorted_intrinsics.json'))
        s.Kfull = np.array([[K['fx'], 0, K['cx']], [0, K['fy'], K['cy']], [0, 0, 1.]]); s.W0, s.H0 = K['width'], K['height']
        d0 = cv2.imread(f'{s.depth_dir}/000000.png', -1); s.Hd, s.Wd = d0.shape; sc = s.Wd / s.W0
        s.Kd = s.Kfull.copy(); s.Kd[:2] *= sc
        s.cm = cammodel.Models(f'{HAW}/episode_{ep:06d}/{s.names[-1]}/anycalib', s.Wd, s.Hd, s.W0)
        Rm, _ = s.cm.rot_new_to_ours(); s.R_n2o = np.eye(4); s.R_n2o[:3, :3] = Rm
        s.cam_mode = 'own'   # 'own': NEW with its official-cam-a model; 'pinhole': NEW treated like OURS (unfair control)
        nd = E + f'/episode_{ep:06d}'; cm = np.load(nd + '/camera_motion.npz'); s.c0Tc = cm['cam0_T_cam']; s.Ta = cm['T_a_from_rectL']
        s.Tnew = {o: np.load(f'{nd}/{o}_trajectory.npz')['T_world_obj'] for o in s.names}
        s.mesh = {('ours', o): trimesh.load(s.fr + f'/assets/{o}_visual.obj', force='mesh') for o in s.names}
        s.mesh.update({('new', o): trimesh.load(f'{nd}/{o}_mesh_metric.obj', force='mesh') for o in s.names})
        s.samp = {k: np.asarray(trimesh.sample.sample_surface(m, nsamp, seed=0)[0]) for k, m in s.mesh.items()}
        s.tree = {k: cKDTree(v) for k, v in s.samp.items()}
    # object pose in RGB camera of frame f
    def cam_T(s, track, o, f):
        if track == 'ours': return np.linalg.inv(s.wfc[f]) @ s.W[f, s.names.index(o)]
        return s.Ta @ np.linalg.inv(s.c0Tc[f]) @ s.Tnew[o][f]
    def depth(s, f):
        d = cv2.imread(f'{s.depth_dir}/{f:06d}.png', -1).astype(np.float64); dep = 65535.0 / d - 1.0; dep[(d >= 65535) | (dep < .1) | (dep > 1.5)] = np.nan; return dep
    def _m(s, p):
        m = cv2.imread(p, -1)
        if m is None: return np.zeros((s.Hd, s.Wd), bool)
        if m.ndim == 3: m = m[..., 0]
        return cv2.resize((m > 0).astype(np.uint8), (s.Wd, s.Hd), interpolation=cv2.INTER_NEAREST) > 0
    def mask(s, o, f): return s._m(f'{s.mask_dir[o]}/{f:06d}.png')
    def hand(s, f):
        h = np.zeros((s.Hd, s.Wd), bool)
        for d in s.hand_dirs: h |= s._m(f'{d}/{f:06d}.png')
        return h
    def zbuf(s, key, C, model):
        """splat z-buffer of the mesh samples posed by camera-from-object C, projected with that track's camera model."""
        X = s.samp[key] @ C[:3, :3].T + C[:3, 3]; X = X[X[:, 2] > 0.05]; uv, zz = s.cm.project(model, X); u = np.round(uv[:, 0]).astype(int); v = np.round(uv[:, 1]).astype(int)
        ok = (u >= 0) & (u < s.Wd) & (v >= 0) & (v < s.Hd); z = np.full(s.Hd * s.Wd, np.inf); np.minimum.at(z, v[ok] * s.Wd + u[ok], zz[ok]); return z.reshape(s.Hd, s.Wd)
    def model_of(s, track):
        if s.ep in (15, 16): return 'new'      # our ep15/16 reference is the earlier compose delivery: official cam-a frame
        return 'new' if (track == 'new' and s.cam_mode == 'own') else 'ours'
    def to_official(s, track, C):
        """express a camera-frame pose in the official cam-a frame (ours legacy: Kout camera -> official via the best-fit rotation)."""
        return C if s.model_of(track) == 'new' else np.linalg.inv(s.R_n2o) @ C
    def evidence(s, track, o, f, dep=None, objm=None, hand=None, others=()):
        """returns dict: dm (data->model median mm, trimmed mean), md (model->data median |dz| mm over rendered∩mask),
        fs (free-space violation fraction of rendered px: stereo > model+15mm), iou (hand px ignored), n points."""
        key = (track, o); C = s.cam_T(track, o, f)
        dep = s.depth(f) if dep is None else dep; objm = s.mask(o, f) if objm is None else objm; hand = s.hand(f) if hand is None else hand
        handd = cv2.dilate(hand.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        me = cv2.erode(objm.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
        sel = me & ~handd & np.isfinite(dep); P = s.cm.backproject(s.model_of(track), sel, dep)
        r = dict(n=int(len(P)))
        if len(P) >= 30:
            loc = (P - C[:3, 3]) @ C[:3, :3]; d = np.sort(s.tree[key].query(loc)[0] * 1000)
            r['dm'] = float(np.median(d)); r['dm_tm'] = float(d[:int(.8 * len(d))].mean()); r['in10'] = float((d < 10).mean()); r['in5'] = float((d < 5).mean())
        z = s.zbuf(key, C, s.model_of(track)); rend = np.isfinite(z)
        rend_e = cv2.erode(rend.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        # occluders: other objects' masks and hands count as possibly in front
        valid = rend_e & np.isfinite(dep) & ~handd
        for om in others: valid &= ~(cv2.dilate(om.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        if valid.sum() >= 30:
            dz = (dep - z)[valid] * 1000; r['fs'] = float((dz > 15).mean()); r['fs_n'] = int(valid.sum())
            vm = valid & objm
            r['md'] = float(np.median(np.abs((dep - z)[vm]) * 1000)) if vm.sum() >= 30 else np.nan
        care = ~handd
        inter = (rend & objm & care).sum(); uni = ((rend | objm) & care).sum(); r['iou'] = float(inter / max(uni, 1))
        r['rend_px'] = int(rend.sum()); r['mask_px'] = int(objm.sum())
        return r
def pose_diff(Ca, Cb, pts):
    """mean point distance (cm) and rotation angle (deg) between two camera-from-object poses of the same body frame."""
    Xa = pts @ Ca[:3, :3].T + Ca[:3, 3]; Xb = pts @ Cb[:3, :3].T + Cb[:3, 3]
    ang = np.degrees(np.linalg.norm(R.from_matrix(Ca[:3, :3].T @ Cb[:3, :3]).as_rotvec()))
    return float(np.linalg.norm(Xa - Xb, axis=1).mean() * 100), float(ang)
