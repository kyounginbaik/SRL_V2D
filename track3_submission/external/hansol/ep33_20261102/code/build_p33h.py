#!/usr/bin/env python3
"""Put the ep33 start roll -6 deg (total -16 deg vs v14, HGAZr6 / HGBZr6) into a reconstruction bundle (GT-free, adopted 2026-11-03).
The bundle's ep33 must already carry the 5-7 s refit (p33g). The refit episode may differ
only in the wooden_spoon rows of frames FIRST..LAST-1 and in the two fit reports (annot_roll.json, smooth_corr_report.json);
everything else in the episode must be byte-identical. The merged mark file replaces ANNOT_MARKS_20261103.json. README
gets a paragraph and manifest.json is updated (changed / added lists, sha256 of every file except manifest.json).
usage: build_p33h.py SRC_BUNDLE FIXED_EPISODE_DIR MARKS_JSON OUT_BUNDLE"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

E = 'episode_000033'
FIRST, LAST = 0, 40
REPORTS = ()
MARKS = 'ROLLCURVE_p33h.json'
NOTE = '''
ep33 start roll -6 deg more (2026-11-03 night, GT-free, user-adopted, HGAZr6 / HGBZr6): the R2 retarget, which tracks the p33f
reconstruction at 1.43 cm / 12-15 deg, failed MP-SR officially (submission 57947230), so the reconstruction itself is off. A kit-scored
sweep against the six officially failing ep33 robots (v26, v29, v31a, K33_H1rest_b2, team HGA_c9, our R2) found that only an extra
start roll of -6 deg or more (f0-22, smoothstep back to 0 by f34, about the long PCA axis through the centre of mass) makes all six
fail; time shifts of +-1/+-2 frames and positive rolls do not. -6 deg more is -16 deg vs v14, the start roll at which the team's
leaderboard sweep had its best ep33 AUC (K33_H5r16). This changes f0, the scene-alignment frame. Only ep33 wooden_spoon stored frames
0-33 change; blue_cup and every other episode are untouched. Record: episode_000033/ROLLCURVE_p33h.json (ROLLCURVE.json = the
earlier HGARF_c curve, unchanged).
'''


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    src, fixd, marks, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]); assert not out.exists(), out
    a, b = np.load(src / E / 'object_poses.npz'), np.load(fixd / 'object_poses.npz')
    names = [str(n) for n in a['names']]; assert names == [str(n) for n in b['names']]
    k = names.index('wooden_spoon'); report = {}
    for key in a.files:
        if key in ('world_from_object', 'camera_from_object'):
            assert np.array_equal(np.delete(a[key], k, axis=1), np.delete(b[key], k, axis=1)), f'{key}: non-spoon rows differ'
            assert np.array_equal(a[key][:FIRST, k], b[key][:FIRST, k]), f'{key}: spoon changed before f{FIRST}'
            assert np.array_equal(a[key][LAST:, k], b[key][LAST:, k]), f'{key}: spoon changed at or after f{LAST}'
            ch = np.where(np.any(a[key][:, k] != b[key][:, k], axis=(1, 2)))[0]
            report[key] = [int(ch.min()), int(ch.max()), int(len(ch))] if len(ch) else []
        else:
            assert np.array_equal(a[key], b[key]), f'{key} differs'
    for f in sorted(fixd.iterdir()):
        if f.name == 'object_poses.npz' or f.name in REPORTS or not f.is_file():
            continue
        g = src / E / f.name
        assert g.exists() and sha(g) == sha(f), f'{f.name}: differs from or missing in the bundle'
    assert all((fixd / r).is_file() for r in REPORTS)
    shutil.copytree(src, out, symlinks=False)
    for f in ('object_poses.npz',) + REPORTS:
        shutil.copy(fixd / f, out / E / f)
    shutil.copy(marks, out / E / MARKS)
    with open(out / 'README.md', 'a') as fh:
        fh.write(NOTE)
    mf = json.loads((out / 'manifest.json').read_text())
    for f in ('object_poses.npz',) + REPORTS:
        if f'{E}/{f}' not in mf['changed'] and f'{E}/{f}' not in mf['added']:
            mf['changed'].append(f'{E}/{f}')
    if f'{E}/{MARKS}' not in mf['added']:
        mf['added'].append(f'{E}/{MARKS}')
    files = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json')
    mf['sha256'] = {f: sha(out / f) for f in files}; (out / 'manifest.json').write_text(json.dumps(mf, indent=1))
    print(json.dumps(dict(out=str(out), changed_spoon_frames=report, files=len(files), npz=mf['sha256'][f'{E}/object_poses.npz'][:12])))


if __name__ == '__main__':
    main()
