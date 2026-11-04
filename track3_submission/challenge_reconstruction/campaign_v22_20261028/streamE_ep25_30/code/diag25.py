import sys, numpy as np, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE_ep25_30/code')
from elib import *
Q = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25'
CC = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates'
names = ['wooden_piece_1', 'wooden_piece_2']
for n in names:
    m = trimesh.load(f'{TGT}/episode_000025/{n}_mesh_metric.obj', force='mesh'); print(n, 'target extents', m.extents)
variants = {'slotC1': f'{CC}/C_ep25_slotC1/episode_000025/meshes/episode_000025/%s.obj',
            'slotless': f'{CC}/Q_ep25_slotfix_w4L47/episode_000025/meshes/episode_000025/%s.obj',
            'slotted_v1': f'{Q}/meshes_slotted/%s_visual_slotted.obj', 'slotted_v2': f'{Q}/meshes_slotted_v2/%s_visual_slotted.obj'}
for vn, pat in variants.items():
    for slot, n in enumerate(names):
        p = pat % n; m = trimesh.load(p, force='mesh')
        X, ca, rot, anchor = registration(25, slot, n, p)
        ds = []
        for s in range(1, 21):
            X2, *_ = registration(25, slot, n, p, seed=s)
            ds.append(np.degrees(np.linalg.norm(R.from_matrix(X[:3, :3].T @ X2[:3, :3]).as_rotvec())))
        ds = np.array(ds)
        print(vn, n, 'ext', np.round(m.extents, 4), 'agree(<10deg) %d/20' % (ds < 10).sum(), 'flip(>150) %d/20' % (ds > 150).sum(), 'rotdeg', round(float(np.degrees(np.linalg.norm(R.from_matrix(rot).as_rotvec()))), 1))
