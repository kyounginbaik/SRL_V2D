"""Fixed-geometry marker test: the same physical spoon (same 3 balls) appears in every episode. Canonical ball body coordinates =
per-ball median of the free per-track fits over all episodes and both tracks (fit_*.json). Each track is then scored with the
canonical balls (no per-track freedom), so constant along-handle offsets are also caught. Per detection: reprojection distance
to the nearest canonical ball, own camera model. usage: canon_markers.py OUT.json EP..."""
import sys, json, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import h2hlib as H, cammodel as CM, meshreg
D = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/h2h_spoon/markers'
out = sys.argv[1]; eps = [int(x) for x in sys.argv[2:]]
allb = []; G = {}; EP = {}
for ep in eps:
    EP[ep] = e = H.Ep(ep, nsamp=10); G[ep] = meshreg.register(e.mesh[('ours', 'wooden_spoon')], e.mesh[('new', 'wooden_spoon')])[0][0]   # p_new = G p_ours
    z = json.load(open(f'{D}/fit_{ep}_wooden_spoon.json'))
    for tr in ('ours', 'new'):
        b = np.array(z[tr]['balls_cm']) / 100
        if tr == 'ours': b = b @ G[ep][:3, :3].T + G[ep][:3, 3]
        allb.append(b[np.argsort(b[:, 0])] * 100)
allb = np.array(allb); canon = np.median(allb, 0) / 100
print('canonical balls (cm, NEW mesh frame):', np.round(canon * 100, 2).tolist(), ' spread (MAD cm):', np.round(np.median(np.abs(allb / 100 - canon), 0) * 100, 2).tolist())
TW = np.diag([1, -1, -1.])
res = dict(canonical_cm_newframe=(canon * 100).tolist(), G={ep: G[ep].tolist() for ep in eps}, eps={})
for ep in eps:
    z = json.load(open(f'{D}/fit_{ep}_wooden_spoon.json')); F = np.array(z['frames']); UV = np.array(z['uv']); e = EP[ep]; r = dict(frames=F.tolist())
    for tr in ('ours', 'new'):
        m = e.model_of(tr); C = np.array([e.cam_T(tr, 'wooden_spoon', f) for f in range(e.N)])
        cb = canon if tr == 'new' else (canon - G[ep][:3, 3]) @ G[ep][:3, :3]
        for lab, Rm in (('', np.eye(3)), ('twin_', TW)):
            Rf = C[F, :3, :3] @ Rm; tf = C[F, :3, 3]; X = np.einsum('nij,kj->nki', Rf, cb) + tf[:, None]
            uv = np.stack([CM.project_full(e.cm, m, X[:, k])[0] for k in range(3)], 1); r[lab + tr] = np.linalg.norm(uv - UV[:, None], axis=2).min(1).round(2).tolist()
    res['eps'][ep] = r
    a, b = np.array(r['ours']), np.array(r['new'])
    print(f'ep{ep:2d} n={len(F)}  ours med {np.median(a):5.1f}px <10 {np.mean(a<10):.2f} <20 {np.mean(a<20):.2f} | new med {np.median(b):5.1f}px <10 {np.mean(b<10):.2f} <20 {np.mean(b<20):.2f} | twin<20 ours {np.mean(np.array(r["twin_ours"])<20):.2f} new {np.mean(np.array(r["twin_new"])<20):.2f}')
json.dump(res, open(out, 'w'))
