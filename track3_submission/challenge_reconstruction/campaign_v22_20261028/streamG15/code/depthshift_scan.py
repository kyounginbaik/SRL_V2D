"""Scoring-only: does the along-ray component (stereo-depth driven, already on the b/c clock) need less shift than lateral/rotation?
For alpha in {0,0.5,1}: camera-frame translation along the ray shifted by alpha*delta, everything else by delta. Usage EP"""
import sys, json, numpy as np, kitx
from retime import shift
ep = int(sys.argv[1]); X = kitx.load_export(ep, 'r8_base')
D = json.load(open('../out/sync_delta.json'))['public_' + str(ep)]
z = np.load(kitx.root(ep) / 'exports/r8_base/object_poses.npz'); C = z['camera_from_object'].astype(float); Wc = z['world_from_rgb'].astype(float)
Wd = shift(Wc, D); base = None; res = {}
for alpha in [1.0, 0.75, 0.5, 0.0]:
    tr = {}
    for i, n in enumerate(X['names']):
        cd = shift(C[:, i], D); ca = shift(C[:, i], alpha * D)
        u = cd[:, :3, 3] / np.linalg.norm(cd[:, :3, 3], axis=1, keepdims=True)
        cd[:, :3, 3] += u * np.sum(u * (ca[:, :3, 3] - cd[:, :3, 3]), 1, keepdims=True)
        tr[n] = Wd @ cd
    o = kitx.score(ep, tr, X['meshes'], regs=base)
    if base is None: base = {n: (o['cas'][s], o['rots'][s]) for s, n in enumerate(o['objs'])}
    res[alpha] = {n: (float(o['pos'][:, s].mean()), float(o['ori'][:, s].mean())) for s, n in enumerate(o['objs'])}
    print(ep, alpha, res[alpha], flush=True)
json.dump(res, open(f'../out/dshift_{ep}.json', 'w'))
