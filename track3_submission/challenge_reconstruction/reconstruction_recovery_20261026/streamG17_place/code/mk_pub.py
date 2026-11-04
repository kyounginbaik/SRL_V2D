"""Make a public export TAG_OUT = g15_camrt + G17 frame-0 corrections from out/ref/<V>/public_<ep>.json.
Usage: mk_pub.py EP V MODE TAG_OUT  (then score_bundle.py)"""
import sys, json, shutil, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from g17lib import eproot
from apply_f0 import apply
ep, V, mode, tag = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
src = eproot('public', ep) / 'exports' / 'g15_camrt'; dst = eproot('public', ep) / 'exports' / tag
if dst.exists(): raise SystemExit('exists')
R = json.loads((Path(__file__).parent.parent / 'out' / 'ref' / V / f'public_{ep}.json').read_text())
dst.mkdir()
for f in src.iterdir():
    if f.is_file() and f.name not in ('public_score.json', 'object_poses.npz'): shutil.copy2(f, dst / f.name)
z, info = apply(np.load(src / 'object_poses.npz'), R, mode)
np.savez_compressed(dst / 'object_poses.npz', **z)
(dst / 'g17.json').write_text(json.dumps(dict(source=str(src), refine=V, mode=mode, info=info, ground_truth_used=False), indent=1)); print(info)
