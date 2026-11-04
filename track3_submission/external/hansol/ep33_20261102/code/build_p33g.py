#!/usr/bin/env python3
"""Put the ep33 spoon roll refit at 5-7 s from the user's f96-150 marks (HGAZ / HGBZ) into a reconstruction bundle (GT-free,
adopted 2026-11-03). The bundle's ep33 must already carry the 9.2-9.5 s flip fix (p33f). The refit episode may differ
only in the wooden_spoon rows of frames FIRST..LAST-1 and in the two fit reports (annot_roll.json, smooth_corr_report.json);
everything else in the episode must be byte-identical. The merged mark file replaces ANNOT_MARKS_20261103.json. README
gets a paragraph and manifest.json is updated (changed / added lists, sha256 of every file except manifest.json).
usage: build_p33g.py SRC_BUNDLE FIXED_EPISODE_DIR MARKS_JSON OUT_BUNDLE"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

E = 'episode_000033'
FIRST, LAST = 85, 156
REPORTS = ('annot_roll.json', 'smooth_corr_report.json')
MARKS = 'ANNOT_MARKS_20261103.json'
NOTE = '''
ep33 spoon roll at 5-7 s from the user's marks (2026-11-03 night, GT-free, user-adopted, HGAZ / HGBZ): the user marked the RGB frames
f96-150 every 2 frames (out / in / edge-on arrow). Edge-on marks (f112-120 concave down-left, f126-132 up-right, f140-142 down) were given
a facing cap of +0.3 (spoon_annot_roll2.py), the window f88-148 was refit (marks, marker balls, silhouette IoU f96-146) and smoothed
(sigma 1.5). Mark agreement f96-150: 19 -> 24 of 28; silhouette IoU 0.609 -> 0.649; marker balls unchanged. The edge-on frames are now
thin as in the RGB (f116 facing +0.59 -> +0.16, f128 +0.84 -> +0.25). Kit-scored, the six officially failing robots v26, v29, v31a,
K33_H1rest_b2 and v33a_ep33HGA_nospin (team HGA_c9) still fail, but our R2 retarget (official MP-SR fail on 11/03) still passes against
this reconstruction (spoon 1.43 cm / 14.6 deg), so a larger whole-episode error remains (start roll or timing, open). Only ep33
wooden_spoon stored frames 92-151 change; f0, blue_cup and every other episode are untouched. Record: episode_000033/annot_roll.json,
smooth_corr_report.json, ANNOT_MARKS_20261103.json (with all caps).
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
