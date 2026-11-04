#!/usr/bin/env python3
"""Put the ep33 spoon roll fix into a reconstruction bundle (GT-free, adopted 2026-11-02 14:45).
episode_000033/object_poses.npz is replaced by the roll-refit one (HGAR8s15 / HGBR8s15: spoon roll refit f165-259 with
facing 0.8 + silhouette term, correction smoothed sigma 1.5 frames on the stored clock, f0 bit-identical). The fixed
episode must match the bundle's own ep33 everywhere except the wooden_spoon rows; report files the fix adds are copied.
README gets a paragraph and manifest.json is updated (changed / added lists, sha256 of every file except manifest.json).
usage: build_p33.py SRC_BUNDLE FIXED_EPISODE_DIR OUT_BUNDLE VARIANT(HGA|HGB)"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

E = 'episode_000033'
FACING = {'HGA': '0.47 / 0.32 / 0.34 -> 0.79 / 0.73 / 0.74', 'HGB': 'about 0.5 / 0.3 / 0.3 -> 0.79 / 0.78 / 0.73'}
NOTE = '''
ep33 spoon roll fix (2026-11-02, GT-free, user-adopted): at 8.8 s (f176) and 12.4 s (f248) the video shows the spoon's concave
side facing the camera, but the track held it rotated sideways (the front/back annotation fit only enforced the sign, facing
>= 0.25). The spoon roll about its long axis is refit over the stored-clock frames f165-259 to facing >= 0.8 with a
silhouette (IoU) term; the correction (body-frame rotation and translation) is smoothed with sigma 1.5 frames, so untouched
frames are bit-identical (f0, the scene-alignment frame, included). Long axis and position are unchanged, so the image
outline barely moves. Facing at f176 / f180 / f248: {facing}. Front/back marks 47/51 unchanged. Only ep33 wooden_spoon
changes; blue_cup and every other episode are untouched. Reports: episode_000033/smooth_corr_report.json and the roll fit
files added below.
'''


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    src, fixd, out, var = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]; assert not out.exists(), out
    a, b = np.load(src / E / 'object_poses.npz'), np.load(fixd / 'object_poses.npz')
    names = [str(n) for n in a['names']]; assert names == [str(n) for n in b['names']], (names, b['names'])
    k = names.index('wooden_spoon'); report = {}
    for key in a.files:
        if key in ('world_from_object', 'camera_from_object'):
            assert np.array_equal(np.delete(a[key], k, axis=1), np.delete(b[key], k, axis=1)), f'{key}: non-spoon rows differ'
            ch = np.where(np.any(a[key][:, k] != b[key][:, k], axis=(1, 2)))[0]
            assert np.array_equal(a[key][0, k], b[key][0, k]), f'{key}: f0 spoon changed'
            report[key] = [int(ch.min()), int(ch.max()), int(len(ch))] if len(ch) else []
        else:
            assert np.array_equal(a[key], b[key]), f'{key} differs'
    # the roll refit rewrites its own fit reports; anything else (camera, meshes, intrinsics, ...) must be byte-identical
    REPORTS = {'annot_roll.json', 'smooth_corr_report.json', 'handle_smooth_fit_report.json', 'contact_smooth_report.json'}
    added, changed = [], []
    for f in sorted(fixd.iterdir()):
        if f.name == 'object_poses.npz' or not f.is_file():
            continue
        g = src / E / f.name
        if not g.exists():
            added.append(f.name)
        elif sha(g) != sha(f):
            assert f.name in REPORTS, f'{f.name}: differs from the bundle'
            changed.append(f.name)
    shutil.copytree(src, out, symlinks=False)
    shutil.copy(fixd / 'object_poses.npz', out / E / 'object_poses.npz')
    for n in added + changed:
        shutil.copy(fixd / n, out / E / n)
    with open(out / 'README.md', 'a') as fh:
        fh.write(NOTE.format(facing=FACING[var]))
    mf = json.loads((out / 'manifest.json').read_text())
    for n in ['object_poses.npz'] + changed:
        if f'{E}/{n}' not in mf['changed']:
            mf['changed'].append(f'{E}/{n}')
    for n in added:
        if f'{E}/{n}' not in mf['added']:
            mf['added'].append(f'{E}/{n}')
    files = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json')
    mf['sha256'] = {f: sha(out / f) for f in files}; (out / 'manifest.json').write_text(json.dumps(mf, indent=1))
    print(json.dumps(dict(out=str(out), changed_spoon_frames=report, reports_changed=changed, added=added, files=len(files),
                          npz=mf['sha256'][f'{E}/object_poses.npz'][:12])))


if __name__ == '__main__':
    main()
