"""K32 A2 (GT-free; copy of K33-recon2 cuptable.py for ep32, both objects): per static frame, fit the table plane LOCALLY from OUR stereo depth
in a ring 6-20 cm around the object (outside cup/spoon masks), camera frame, and report the object mesh's lowest point relative to it (mm).
usage: table32.py POSES.npz MESHDIR OBJ f0,f1,..   (POSES = object_poses.npz format with camera_from_object)"""
import sys, json, numpy as np, cv2, trimesh
PZ, MD, OBJ = sys.argv[1], sys.argv[2], sys.argv[3]; F = [int(x) for x in sys.argv[4].split(',')]
C = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000032'
K = json.load(open(f'{C}/intrinsics.json')); fx, fy, cx, cy, W, H = K['fx'], K['fy'], K['cx'], K['cy'], K['width'], K['height']
z = np.load(PZ, allow_pickle=False); i = list(z['names']).index(OBJ); Tc = z['camera_from_object'][:, i]
v = np.asarray(trimesh.load(f'{MD}/{OBJ}.obj', force='mesh').vertices)
uu, vv = np.meshgrid(np.arange(W), np.arange(H))
def rd(p): return cv2.imread(p, cv2.IMREAD_UNCHANGED)
hs = []
for t in F:
    occ = np.zeros((H, W), bool)
    for o in ['blue_cup', 'wooden_spoon']:
        for k in '123':
            m = rd(f'{C}/objects/{o}/masks/{k}/{t:06d}.png')
            if m is not None: occ |= m > 0
    occ = cv2.dilate(occ.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    dv = rd(f'{C}/depth/{t:06d}.png').astype(np.float64); D = np.where(dv < 65535, 65535 / np.maximum(dv, 1) - 1, np.nan)
    P = np.stack([(uu - cx) / fx * D, (vv - cy) / fy * D, D], -1)
    pv = v @ Tc[t][:3, :3].T + Tc[t][:3, 3]; ctr = pv.mean(0)
    dist = np.linalg.norm(P - ctr, axis=-1); sel = (~occ) & np.isfinite(D) & (dist > 0.06) & (dist < 0.20); Q = P[sel]
    best = None; rng = np.random.default_rng(0)
    for _ in range(400):
        s = Q[rng.choice(len(Q), 3, replace=False)]; n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n); inl = np.abs((Q - s[0]) @ n) < 0.004
        if best is None or inl.sum() > best[0].sum(): best = (inl, n, s[0])
    q = Q[best[0]]; qc = q.mean(0); n = np.linalg.svd(q - qc)[2][2]
    if n @ (-qc) < 0: n = -n
    h = (pv - qc) @ n; hs.append(1000 * h.min())
    print(f'{OBJ} f{t}: n_inl={best[0].sum()} rms={1000*np.std((q-qc)@n):.1f}mm lowest vs local plane {1000*h.min():+.1f}mm', flush=True)
print(f'SUMMARY {OBJ} {PZ.split("/")[-3:]} median lowest {np.median(hs):+.1f}mm', flush=True)
