"""Camera models of the two tracks onto OUR evidence images (AnyCalib-undistorted RGB, depth_rgb at half resolution).
OURS: pinhole Kout (the camera our tracker and depth_rgb back-projection use).
NEW : official ego_cam_a frame (their T_a_from_rectL); a 3D point maps to our image exactly like project_depth.py placed the
      stereo points: official K/D projection to the raw image, then the AnyCalib fisheye undistortion to Kout.
Depth_rgb stores official-cam-a z at those pixels, so NEW is compared with z directly; OURS back-projects with Kout (as our
tracker did)."""
import json, numpy as np, cv2
CAL = json.load(open('/data1/video_to_data_challenge/track_3/public/meta/camera_calibration.json'))['cameras']['ego_cam_a']
Koff = np.array(CAL['intrinsic_matrix'], float); Doff = np.array(CAL['distortion_coefficients'], float)
def _mat(p):
    a = json.load(open(p)); return np.array([[a['fx'], 0, a['cx']], [0, a['fy'], a['cy']], [0, 0, 1.]])
class Models:
    def __init__(s, anycalib_dir, Wd, Hd, W0=2028):
        s.Kany = _mat(anycalib_dir + '/intrinsics.json'); s.Dany = np.array(json.load(open(anycalib_dir + '/distortion.json'))['params'], float)
        s.Kout = _mat(anycalib_dir + '/undistorted_intrinsics.json'); s.sc = Wd / W0; s.Kd = s.Kout.copy(); s.Kd[:2] *= s.sc; s.Wd, s.Hd = Wd, Hd
        v, u = np.indices((Hd, Wd)); s.pix = np.stack([u.ravel(), v.ravel()], 1).astype(np.float64)
        s.ray = {'ours': np.stack([(s.pix[:, 0] - s.Kd[0, 2]) / s.Kd[0, 0], (s.pix[:, 1] - s.Kd[1, 2]) / s.Kd[1, 1]], 1)}
        # NEW: our pixel -> normalized Kout -> fisheye distort -> raw px -> official undistort -> normalized official ray
        nrm = s.ray['ours'].reshape(-1, 1, 2)
        raw = cv2.fisheye.distortPoints(nrm, s.Kany, s.Dany).reshape(-1, 2)
        s.ray['new'] = cv2.undistortPoints(raw.reshape(-1, 1, 2), Koff, Doff).reshape(-1, 2)
    def project(s, track, X):
        """X (n,3) camera-frame points of that track's camera -> (n,2) pixel coords at depth resolution, z."""
        z = X[:, 2]
        if track == 'ours': return np.stack([X[:, 0] / z * s.Kd[0, 0] + s.Kd[0, 2], X[:, 1] / z * s.Kd[1, 1] + s.Kd[1, 2]], 1), z
        raw = cv2.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), Koff, Doff)[0]
        uv = cv2.fisheye.undistortPoints(raw, s.Kany, s.Dany, P=s.Kd).reshape(-1, 2); return uv, z
    def backproject(s, track, mask, dep):
        idx = np.flatnonzero(mask.ravel()); Z = dep.ravel()[idx]; r = s.ray[track][idx]; return np.stack([r[:, 0] * Z, r[:, 1] * Z, Z], 1)
    def rot_new_to_ours(s):
        """best rotation R with ray_ours ~ R ray_new (camera-frame change official-a -> Kout camera) and the residual (deg)."""
        a = np.c_[s.ray['new'], np.ones(len(s.pix))]; b = np.c_[s.ray['ours'], np.ones(len(s.pix))]
        sel = np.arange(0, len(a), 97); a = a[sel] / np.linalg.norm(a[sel], axis=1, keepdims=True); b = b[sel] / np.linalg.norm(b[sel], axis=1, keepdims=True)
        U, S_, Vt = np.linalg.svd(a.T @ b); Rm = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        res = np.degrees(np.arccos(np.clip(np.sum((a @ Rm.T) * b, 1), -1, 1))); return Rm, res
def rays_at(models, model, uv_full):
    """normalized camera rays (x, y, 1) of full-resolution evidence-image pixels for a camera model."""
    uv_full = np.asarray(uv_full, float).reshape(-1, 2); K = models.Kout
    nrm = np.stack([(uv_full[:, 0] - K[0, 2]) / K[0, 0], (uv_full[:, 1] - K[1, 2]) / K[1, 1]], 1)
    if model == 'new':
        raw = cv2.fisheye.distortPoints(nrm.reshape(-1, 1, 2), models.Kany, models.Dany).reshape(-1, 2)
        nrm = cv2.undistortPoints(raw.reshape(-1, 1, 2), Koff, Doff).reshape(-1, 2)
    return np.c_[nrm, np.ones(len(nrm))]
def project_full(models, model, X):
    uv, z = models.project(model, X); return uv / models.sc, z
