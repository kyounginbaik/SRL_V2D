"""K33-recon2 (GT-free): on static-cup frames fit the table plane LOCALLY from OUR stereo depth in a ring around the cup (outside cup/hand/
spoon masks, within 6-20 cm of the cup axis), in the camera frame, and report the cup mesh's lowest point relative to it (mm), the cup
axis tilt vs plane normal, and the global support-plane (world z=0) offset at the same place. usage: cuptable.py BUNDLE f0,f1,.."""
import sys, json, numpy as np, cv2, trimesh
B = sys.argv[1]; F = [int(x) for x in sys.argv[2].split(',')]
C = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033'
K = json.load(open(f'{C}/intrinsics.json')); fx, fy, cx, cy, W, H = K['fx'], K['fy'], K['cx'], K['cy'], K['width'], K['height']
z = np.load(f'{B}/object_poses.npz', allow_pickle=False); i = list(z['names']).index('blue_cup'); Tc = z['camera_from_object'][:, i]; Wr = z['world_from_rgb']
m = trimesh.load(f'{B}/meshes/blue_cup.obj', force='mesh'); v = np.asarray(m.vertices)
c0 = v - v.mean(0); w, U = np.linalg.eigh(c0.T @ c0 / len(c0)); ax = U[:, 2]   # cup long axis
uu, vv = np.meshgrid(np.arange(W), np.arange(H))
def rd(p): return cv2.imread(p, cv2.IMREAD_UNCHANGED)
for t in F:
    occ = np.zeros((H, W), bool)
    for o in ['blue_cup', 'wooden_spoon']:
        for k in '123': occ |= rd(f'{C}/objects/{o}/masks/{k}/{t:06d}.png') > 0
    occ = cv2.dilate(occ.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    dv = rd(f'{C}/depth/{t:06d}.png').astype(np.float64); D = np.where(dv < 65535, 65535 / np.maximum(dv, 1) - 1, np.nan)
    P = np.stack([(uu - cx) / fx * D, (vv - cy) / fy * D, D], -1)
    pv = v @ Tc[t][:3, :3].T + Tc[t][:3, 3]; a = Tc[t][:3, :3] @ ax; ctr = pv.mean(0)
    rel = P - ctr; radial = np.linalg.norm(rel - (rel @ a)[..., None] * a, axis=-1)
    sel = (~occ) & np.isfinite(D) & (radial > 0.06) & (radial < 0.20)
    Q = P[sel]
    best = None; rng = np.random.default_rng(0)
    for _ in range(400):
        s = Q[rng.choice(len(Q), 3, replace=False)]; n = np.cross(s[1] - s[0], s[2] - s[0]); 
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n); inl = np.abs((Q - s[0]) @ n) < 0.004
        if best is None or inl.sum() > best[0].sum(): best = (inl, n, s[0])
    inl = best[0]; q = Q[inl]; qc = q.mean(0); n = np.linalg.svd(q - qc)[2][2]
    if n @ (-qc) < 0: n = -n                          # normal toward camera (up)
    h = (pv - qc) @ n
    Wn = Wr[t][:3, :3].T @ np.array([0, 0, 1.0]); Wo = (np.linalg.inv(Wr[t]) @ np.array([0, 0, 0, 1.0]))[:3]
    hw = (pv - Wo) @ Wn
    tilt = np.degrees(np.arccos(min(1, abs(a @ n)))); ang = np.degrees(np.arccos(min(1, abs(n @ Wn))))
    print(f'f{t}: local plane n_inl={inl.sum()} rms={1000*np.std((q-qc)@n):.1f}mm | cup lowest vs LOCAL plane {1000*h.min():+.1f}mm, vs WORLD z=0 {1000*hw.min():+.1f}mm | cup axis tilt vs local {tilt:.1f} deg | local vs world normal {ang:.2f} deg, offset at cup {1000*((qc-Wo)@Wn):+.1f}mm')
