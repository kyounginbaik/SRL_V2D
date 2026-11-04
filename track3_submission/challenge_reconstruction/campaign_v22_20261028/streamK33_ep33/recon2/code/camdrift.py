"""K33-recon2 (GT-free) camera-trajectory check: static background (table+floor, outside object/hand masks) must stay fixed in world.
For each frame t and reference frames r: ORB matches t<->r on static pixels, back-project with OUR stereo depth, map through the bundle's
world_from_rgb, report the median world displacement (cm) of matched static points = camera pose error between t and r.
usage: camdrift.py BUNDLE_EP_DIR OUT.tsv"""
import sys, os, json, numpy as np, cv2
from multiprocessing import Pool
B, OUT = sys.argv[1], sys.argv[2]
C = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033'
K = json.load(open(f'{C}/intrinsics.json')); fx, fy, cx, cy, W, H = K['fx'], K['fy'], K['cx'], K['cy'], K['width'], K['height']
Wr = np.load(f'{B}/object_poses.npz', allow_pickle=False)['world_from_rgb']
cap = cv2.VideoCapture(f'{C}/rgb.mp4'); FR = []
while True:
    ok, im = cap.read()
    if not ok: break
    FR.append(cv2.cvtColor(cv2.resize(im, (W, H)), cv2.COLOR_BGR2GRAY))
REFS = [0, 60, 120, 180, 240, 300, 347]
def rd(p): return cv2.imread(p, cv2.IMREAD_UNCHANGED)
def static(t):
    m = np.zeros((H, W), bool)
    for o in ['blue_cup', 'wooden_spoon']:
        for k in '123': m |= rd(f'{C}/objects/{o}/masks/{k}/{t:06d}.png') > 0
    m = cv2.dilate(m.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    dv = rd(f'{C}/depth/{t:06d}.png').astype(np.float64); D = np.where(dv < 65535, 65535 / np.maximum(dv, 1) - 1, np.nan)
    D[(D < 0.2) | (D > 2.5)] = np.nan
    return ~m, D
cv2.setNumThreads(0)
def feats(t):
    s, D = static(t); kp, de = cv2.ORB_create(3000).detectAndCompute(FR[t], s.astype(np.uint8) * 255); return kp, de, D
def world(kp, D, t, idx):
    uv = np.array([kp[i].pt for i in idx]); u, v = uv[:, 0], uv[:, 1]; z = D[v.astype(int), u.astype(int)]
    p = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z, np.ones_like(z)], 1); return (p @ Wr[t].T)[:, :3]
RF = {}
def one(t):
    kt, dt, Dt = feats(t); out = [t]
    for r in REFS:
        kr, dr, Dr = RF[r]
        if dt is None or dr is None or r == t: out += [np.nan, 0]; continue
        ms = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(dt, dr, k=2)
        g = [a for a, b in (x for x in ms if len(x) == 2) if a.distance < 0.75 * b.distance]
        if len(g) < 15: out += [np.nan, len(g)]; continue
        pt = np.array([kt[m.queryIdx].pt for m in g]); pr = np.array([kr[m.trainIdx].pt for m in g])
        F, inl = cv2.findFundamentalMat(pt, pr, cv2.FM_RANSAC, 1.5, 0.999)
        if inl is None: out += [np.nan, 0]; continue
        g = [m for m, i in zip(g, inl.ravel()) if i]
        a = world(kt, Dt, t, [m.queryIdx for m in g]); b = world(kr, Dr, r, [m.trainIdx for m in g])
        ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
        if ok.sum() < 10: out += [np.nan, int(ok.sum())]; continue
        d = np.linalg.norm(a[ok] - b[ok], axis=1); out += [float(np.median(d) * 100), int(ok.sum())]
    return out
for r in REFS: RF[r] = feats(r)
if __name__ == '__main__':
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as P: rows = P.map(one, range(len(Wr)))
    hdr = 'frame\t' + '\t'.join(f'd{r}_cm\tn{r}' for r in REFS)
    open(OUT, 'w').write(hdr + '\n' + '\n'.join('\t'.join(f'{x:.3f}' if isinstance(x, float) else str(x) for x in r) for r in rows) + '\n')
    print('done', len(rows))
