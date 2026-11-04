import json, numpy as np, sys
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamV_C19/code')
from common19 import *
packed = packed_meshes(); reg = registration(packed); comp = compose_tracks()
man = json.loads((EPDIR / 'input_manifest.json').read_text()); S = np.array(man['setup_from_source'])
d = np.load(SRC / 'object_poses.npz'); W = d['world_from_object'].astype(float)
ours = {n: np.einsum('ij,tjk->tik', S, W[:, i]) for i, n in enumerate(NAMES)}; cb = {n: comp[n] @ reg[n]['M'] for n in NAMES}
for lab, P in (('ours', ours), ('compose', cb)):
    for n in NAMES:
        p = P[n][:, :3, 3]; disp = np.linalg.norm(p - p[0], axis=1) * 100
        rot = [np.degrees(R.from_matrix(P[n][0, :3, :3].T @ P[n][t, :3, :3]).magnitude()) for t in range(235)]
        print(lab, n, 'disp cm per 0.5s', disp[::10].round(1).tolist()); print('   rot', np.round(rot[::10], 0).tolist())
