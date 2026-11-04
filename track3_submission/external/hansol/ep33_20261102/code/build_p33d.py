#!/usr/bin/env python3
"""Put the ep33 spoon roll refit from the user's 11/03 front/back marks (HGAYs15 / HGBYs15) into a reconstruction bundle
(GT-free, adopted 2026-11-03). The bundle's ep33 must already carry HGARF_c / HGBRF_c (p33c). The refit episode may differ
only in the wooden_spoon rows of frames FIRST..LAST-1 and in the two fit reports (annot_roll.json, smooth_corr_report.json);
everything else in the episode must be byte-identical. The merged mark file is added as ANNOT_MARKS_20261103.json. README
gets a paragraph and manifest.json is updated (changed / added lists, sha256 of every file except manifest.json).
usage: build_p33d.py SRC_BUNDLE FIXED_EPISODE_DIR MARKS_JSON OUT_BUNDLE"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

E = 'episode_000033'
FIRST, LAST = 140, 300
REPORTS = ('annot_roll.json', 'smooth_corr_report.json')
MARKS = 'ANNOT_MARKS_20261103.json'
NOTE = '''
ep33 spoon roll from front/back marks (2026-11-03, GT-free, user-adopted, HGAYs15 / HGBYs15): the user marked on the RGB
frames, every 2 frames over f176-224 and f240-280, whether the concave side of the spoon bowl faces the camera (out), faces
away (in), or is seen edge-on (image direction of the concave normal). Merged with the 11/02 marks elsewhere (78 marks), the
spoon roll about its long axis was refit by Viterbi on a 10 deg grid (marks, marker balls, silhouette IoU against the SAM2
mask, smoothness; windows f150-230 and f236-290, zero correction at both window ends), then Gaussian-smoothed (sigma 1.5
frames). Marks in agreement over f150-290: 53/55 (was 50/55); f192-198 now show the concave side to the camera as marked.
All four officially failing robots (v26, v29, v31a, K33_H1rest_b2) still fail against this reconstruction with the kit
scorer. Only ep33 wooden_spoon frames 140-299 change; f0 (scene alignment), blue_cup and every other episode are untouched.
Record: episode_000033/annot_roll.json, smooth_corr_report.json, ANNOT_MARKS_20261103.json.
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
