"""Build selected20_final_v11_g14 = selected20_final_v10_g12 with the G14 hedged world_from_object (camera_from_object re-derived) for every
episode whose hedge differs from v10. Meshes, cameras, everything else untouched. Never overwrites.
usage: build_v11.py TAG OUT_BUNDLE"""
import sys, json, shutil, numpy as np
from pathlib import Path
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026'); TAG, OUT = sys.argv[1], Path(sys.argv[2])
E = Path(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG14/out/eval_{TAG}')
assert not OUT.exists(), f'{OUT} exists (never overwrite)'
shutil.copytree(RR / 'selected20_final_v10_g12', OUT)
changed = {}
for f in sorted(E.glob('ep*.npz')):
    ep = int(f.stem[2:]); h = np.load(f); fol = OUT / f'episode_{ep:06d}'
    d = dict(np.load(fol / 'object_poses.npz')); assert [str(x) for x in d['names']] == [str(x) for x in h['names']]
    dif = np.abs(h['world_from_object'] - d['world_from_object']).max()
    if dif < 1e-9: continue
    d['world_from_object'] = h['world_from_object']; d['camera_from_object'] = np.linalg.inv(d['world_from_rgb'])[:, None] @ h['world_from_object']
    np.savez(fol / 'object_poses.npz', **d)
    info = json.load(open(E / f'ep{ep}.json'))
    (fol / 'g14_hedge.json').write_text(json.dumps(dict(rule=f'G14 hedge {TAG}', params=info['params'], sources=info['info']['sources'],
        admitted=info['admitted'], objects=info['info']['objects']), indent=1, default=str))
    changed[ep] = {n: (round(o['move_cm_mean'], 3), round(o['move_deg_mean'], 2)) for n, o in info['info']['objects'].items()}
(OUT / 'G14_CHANGES.json').write_text(json.dumps(dict(base='selected20_final_v10_g12', rule=f'G14 hedge {TAG}', changed_time_mean_move_cm_deg=changed), indent=1))
print('built', OUT, 'changed', sorted(changed))
