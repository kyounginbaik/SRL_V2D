"""Camera ego-motion by SIFT feature matching + metric stereo depth + PnP RANSAC (robust to moving people/arms; no GT).
Frame 0 keypoints are lifted to 3D with the stereo depth (hand / moving-object masks excluded); every frame k is registered to
frame 0 (fallback: to the last good keyframe, composed). Output cam0_from_camk, world_from_rgb = world_from_cam0(stage A) @ it.
usage: pnp_egomotion.py EP [--mask-dirs DIR ...] [--out NAME]"""
import os, sys, json, argparse, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
pass
ap = argparse.ArgumentParser(); ap.add_argument('ep', type=int); ap.add_argument('--mask-dirs', nargs='*', default=[]); ap.add_argument('--out', default='egomotion_pnp.npz')
ap.add_argument('--video'); ap.add_argument('--depth'); ap.add_argument('--intr'); ap.add_argument('--world0'); ap.add_argument('--hand-dirs', nargs='*'); ap.add_argument('--n', required=True); ap.add_argument('--world0-key', default='world_from_rgb')
a = ap.parse_args(); ep = a.ep; W = None
n = int(a.n); s0 = None
I = json.load(open(a.intr)); sc = 0.5 * (1014.0 / (2 * I['cx'])) if False else 0.5; K = np.array([[I['fx'] * sc, 0, I['cx'] * sc], [0, I['fy'] * sc, I['cy'] * sc], [0, 0, 1]])
D = a.depth; masks = (a.hand_dirs or []) + a.mask_dirs
cap = cv2.VideoCapture(a.video); sift = cv2.SIFT_create(nfeatures=6000); bf = cv2.BFMatcher(cv2.NORM_L2)
def frame(k):
    ok, img = cap.read(); g = cv2.cvtColor(cv2.resize(img, (1014, 760)), cv2.COLOR_BGR2GRAY)
    bad = np.zeros((760, 1014), np.uint8)
    for md in masks:
        m = cv2.imread(f'{md}/{k:06d}.png', 0)
        if m is not None: bad |= cv2.resize((m > 127).astype(np.uint8), (1014, 760), interpolation=cv2.INTER_NEAREST)
    bad = cv2.dilate(bad, np.ones((31, 31), np.uint8)); kp, de = sift.detectAndCompute(g, (1 - bad) * 255)
    d = cv2.imread(f'{D}/{k:06d}.png', -1).astype(float); z = np.where(d < 65535, 65535.0 / np.maximum(d, 1) - 1, np.nan)
    return kp, de, z
def lift(kp, z):
    pts = np.array([k.pt for k in kp]); zz = z[np.clip(pts[:, 1].round().astype(int), 0, 759), np.clip(pts[:, 0].round().astype(int), 0, 1013)]
    X = np.stack([(pts[:, 0] - K[0, 2]) / K[0, 0] * zz, (pts[:, 1] - K[1, 2]) / K[1, 1] * zz, zz], 1); return X, np.isfinite(zz) & (zz > 0.15) & (zz < 4.0)
def pnp(kf, kp, de):
    kpf, def_, Xf, okf = kf
    m = bf.knnMatch(de, def_, k=2); good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance and okf[x[0].trainIdx]]
    if len(good) < 12: return None, 0
    obj = Xf[[g.trainIdx for g in good]]; img = np.array([kp[g.queryIdx].pt for g in good])
    ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=2.0, iterationsCount=2000, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 12: return None, 0
    rv, tv = cv2.solvePnPRefineLM(obj[inl[:, 0]], img[inl[:, 0]], K, None, rv, tv)
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rv)[0]; T[:3, 3] = tv[:, 0]; return np.linalg.inv(T), len(inl)   # kf_cam_from_cam_k
kp0, de0, z0 = frame(0); X0, ok0 = lift(kp0, z0); kf = (kp0, de0, X0, ok0); kf_T = np.eye(4); kf_idx = 0
out = np.tile(np.eye(4), (n, 1, 1)); inl = np.zeros(n, int); inl[0] = len(kp0)
for k in range(1, n):
    kp, de, z = frame(k); T, ni = pnp(kf, kp, de)
    if T is None or ni < 40:     # try frame 0 directly as a fallback, else hold
        T0, n0 = pnp(((kp0, de0, X0, ok0)), kp, de)
        if T0 is not None and n0 >= max(ni, 40): T, ni, base = T0, n0, np.eye(4)
        else: base = kf_T
        if T is None: out[k] = out[k - 1]; inl[k] = 0; continue
    else: base = kf_T
    out[k] = base @ T; inl[k] = ni
    if ni < 150 and k - kf_idx > 5:   # refresh the keyframe when overlap with it decays
        X, okk = lift(kp, z); kf = (kp, de, X, okk); kf_T = out[k]; kf_idx = k
    if k % 50 == 0: print(k, 'move cm', np.round(out[k][:3, 3] * 100, 1), 'inliers', ni, 'kf', kf_idx, flush=True)
Wc0 = np.load(a.world0)[a.world0_key][0]; Wc = np.einsum('ij,njk->nik', Wc0, out)
os.makedirs(os.path.dirname(a.out), exist_ok=True); np.savez_compressed(a.out, cam0_from_camk=out, inliers=inl, world_from_rgb=Wc)
print('saved; median inliers', int(np.median(inl[1:])), 'frames with <40 inliers', int((inl[1:] < 40).sum()))
