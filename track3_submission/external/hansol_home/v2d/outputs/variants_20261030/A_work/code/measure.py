"""GT-free motion statistics: eval bundle (v8_g10b or a variant) vs public/held-out r8_base exports.
usage: measure.py OUT_JSON [EVAL_BUNDLE] [PUB_TAG]"""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import slib

RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
PUB = Path('/home/hansol/v2d/outputs/variants_20261030/A_work/pub')
out = Path(sys.argv[1])
bundle = Path(sys.argv[2]) if len(sys.argv) > 2 else RR / 'selected20_final_v8_g10b'
ptag = sys.argv[3] if len(sys.argv) > 3 else 'r8_base'


def clip_stats(npz, meshes):
    d = np.load(npz)
    names = [str(n) for n in d['names']]
    res = {}
    for k, n in enumerate(names):
        info = slib.mesh_info(n, meshes[n])
        res[n] = slib.summarize(slib.motion_stats(d['world_from_object'][:, k], info))
        res[n]['frames'] = int(len(d['world_from_object']))
    return res


R = dict(eval={}, public={})
for ep in sorted(bundle.glob('episode_*')):
    mj = json.loads((ep / 'meshes.json').read_text())
    R['eval'][ep.name] = clip_stats(ep / 'object_poses.npz', {n: str(ep / v['file']) for n, v in mj.items()})
for split in ['public', 'public_heldout']:
    for ep in sorted((PUB / split).glob('episode_*')):
        f = ep / 'exports' / ptag
        if not (f / 'object_poses.npz').exists():
            continue
        R['public'][f'{split}/{ep.name}'] = clip_stats(f / 'object_poses.npz', json.loads((f / 'meshes.json').read_text()))
out.write_text(json.dumps(R, indent=1))

keys = ['rot_gt30', 'rot_gt15', 'pos_gt3', 'pos_gt2', 'rot_max', 'rot_p99', 'rot_p90', 'pos_max', 'pos_p99', 'racc_p90', 'racc_gt20', 'pacc_p90', 'pacc_gt1']
print('clip object ' + ' '.join(keys))
for part in ['eval', 'public']:
    tot = {k: 0 for k in keys if 'gt' in k}
    for c, objs in R[part].items():
        for n, s in objs.items():
            print(f"{part[:3]} {c[-8:]:8s} {n:18s} " + ' '.join(f"{s[k]:.1f}" if isinstance(s[k], float) else str(s[k]) for k in keys))
            for k in tot:
                tot[k] += s[k]
    print(part, 'TOTAL', tot)
