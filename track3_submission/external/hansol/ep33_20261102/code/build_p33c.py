#!/usr/bin/env python3
"""Put the ep33 spoon roll curve HGARF_c / HGBRF_c into a reconstruction bundle (GT-free, adopted 2026-11-03).
The bundle's ep33 must already carry the 11/02 roll fix (HGAR8s15 / HGBR8s15, installed as p33). The fixed episode
(ep33_rollcurve_target.py "0:-10,22:-10,34:6,52:6,64:0" on that episode) may differ only in the wooden_spoon rows of frames
0..69; everything else in the episode must be byte-identical, ROLLCURVE.json is added. README gets a paragraph and
manifest.json is updated (changed / added lists, sha256 of every file except manifest.json).
usage: build_p33c.py SRC_BUNDLE FIXED_EPISODE_DIR OUT_BUNDLE"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

E = 'episode_000033'
LAST = 70
NOTE = '''
ep33 spoon start roll curve (2026-11-03, GT-free, user-adopted, HGARF_c / HGBRF_c): the spoon lies still on the board until
the right hand picks it up (about f32). Over f0-24 the silhouette IoU against the SAM2 mask and the marker-ball distance are
best for a roll of -12..-4 deg about the spoon's long axis, and the roll at which the rigid mesh lies lowest on the measured
stereo table plane is -8..-12 deg (median -11). After the grasp (f32-55) the silhouette prefers about +6 deg. The roll
correction is -10 deg over f0-22, a smoothstep to +6 deg at f34, +6 deg to f52, and a smoothstep back to 0 at f64 (long PCA
axis through the centre of mass). -10 is also the smallest start roll for which all four officially failing robots (v26,
v29, v31a, K33_H1rest_b2) fail against this reconstruction with the kit scorer (v31a spoon 3.18 cm). This changes f0, the
scene-alignment frame. Only ep33 wooden_spoon frames 0-63 change; blue_cup and every other episode are untouched.
Record: episode_000033/ROLLCURVE.json.
'''


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    src, fixd, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); assert not out.exists(), out
    a, b = np.load(src / E / 'object_poses.npz'), np.load(fixd / 'object_poses.npz')
    names = [str(n) for n in a['names']]; assert names == [str(n) for n in b['names']]
    k = names.index('wooden_spoon'); report = {}
    for key in a.files:
        if key in ('world_from_object', 'camera_from_object'):
            assert np.array_equal(np.delete(a[key], k, axis=1), np.delete(b[key], k, axis=1)), f'{key}: non-spoon rows differ'
            assert np.array_equal(a[key][LAST:, k], b[key][LAST:, k]), f'{key}: spoon changed at or after f{LAST}'
            ch = np.where(np.any(a[key][:, k] != b[key][:, k], axis=(1, 2)))[0]
            report[key] = [int(ch.min()), int(ch.max()), int(len(ch))] if len(ch) else []
        else:
            assert np.array_equal(a[key], b[key]), f'{key} differs'
    added = []
    for f in sorted(fixd.iterdir()):
        if f.name == 'object_poses.npz' or not f.is_file():
            continue
        g = src / E / f.name
        if g.exists():
            assert sha(g) == sha(f), f'{f.name}: differs from the bundle'
        else:
            added.append(f.name)
    assert added == ['ROLLCURVE.json'], added
    shutil.copytree(src, out, symlinks=False)
    shutil.copy(fixd / 'object_poses.npz', out / E / 'object_poses.npz'); shutil.copy(fixd / 'ROLLCURVE.json', out / E / 'ROLLCURVE.json')
    with open(out / 'README.md', 'a') as fh:
        fh.write(NOTE)
    mf = json.loads((out / 'manifest.json').read_text())
    if f'{E}/object_poses.npz' not in mf['changed']:
        mf['changed'].append(f'{E}/object_poses.npz')
    if f'{E}/ROLLCURVE.json' not in mf['added']:
        mf['added'].append(f'{E}/ROLLCURVE.json')
    files = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json')
    mf['sha256'] = {f: sha(out / f) for f in files}; (out / 'manifest.json').write_text(json.dumps(mf, indent=1))
    print(json.dumps(dict(out=str(out), changed_spoon_frames=report, added=added, files=len(files), npz=mf['sha256'][f'{E}/object_poses.npz'][:12])))


if __name__ == '__main__':
    main()
