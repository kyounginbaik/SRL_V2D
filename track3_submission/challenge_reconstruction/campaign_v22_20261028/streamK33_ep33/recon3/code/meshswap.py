"""K33-recon3 cheap test (GT-free, poses untouched): replace ONLY the ep33 wooden_spoon mesh payload (slot o1) of a base package with the
ep15 wooden_spoon payload of the same package (same G18 spoon, a different tessellation, same body frame: chamfer 1.07 mm, centroid
offset 1.3 mm). The kit registers our mesh to the hidden reference with a seeded multi-start ICP on 3000 sampled surface points; if
ep33's failure is a registration-basin artefact for this seed, a different tessellation of the same shape can change it.
usage: meshswap.py BASE_ALL20 OUT_ALL20"""
import sys, json, shutil, hashlib, numpy as np, pandas as pd
from pathlib import Path
B, O = Path(sys.argv[1]), Path(sys.argv[2]); assert not O.exists(); O.mkdir(parents=True)
shutil.copytree(B / 'submission_kit', O / 'submission_kit', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
(O / 'results').symlink_to((B / 'results').resolve())
s = pd.read_parquet(B / 'track3_submission.parquet'); orig = s.copy()
src = s.index[s.row_id == 'track_3/e000015/w000/f000000/o1']; dst = s.index[s.row_id == 'track_3/e000033/w000/f000000/o1']
assert len(src) == 1 and len(dst) == 1
s.at[dst[0], 'object_mesh_b64'] = s.at[src[0], 'object_mesh_b64']
diff = (s != orig) & ~(s.isna() & orig.isna())
assert diff.columns[diff.any()].tolist() == ['object_mesh_b64'] and s.row_id[diff.any(axis=1)].tolist() == ['track_3/e000033/w000/f000000/o1']
s.to_parquet(O / 'track3_submission.parquet', index=False)
sha = hashlib.sha256((O / 'track3_submission.parquet').read_bytes()).hexdigest()
(O.parent / 'K33_PACKAGE.json').write_text(json.dumps(dict(diagnostic_only=True, label=O.parent.name, episode=33, base=str(B),
    base_sha=hashlib.sha256((B / 'track3_submission.parquet').read_bytes()).hexdigest(), change='ep33 o1 mesh payload := ep15 o1 payload (same G18 spoon, other tessellation); poses untouched',
    sha=sha), indent=1))
(O.parent / 'README.md').write_text('# DIAGNOSTIC ep33 spoon mesh-tessellation swap (registration-basin test)\n\nBase = v29 + ep35 fix (official 0.90). Only the ep33 '
    'wooden_spoon mesh payload changes (ep15 payload of the same G18 spoon, same body frame, chamfer 1.07 mm). Poses untouched.\n'
    'Upload with --only maniptrans_sr: 0.95 -> ep33 failure was a registration-basin artefact; 0.90 -> not.\n')
print(sha)
