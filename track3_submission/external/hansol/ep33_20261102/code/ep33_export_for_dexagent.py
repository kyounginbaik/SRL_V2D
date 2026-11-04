#!/usr/bin/env python3
"""Export one reconstructed episode for the DexAgent converter (runs on the lab server, where the clock code lives).
Writes one npz with everything on the RGB clock (the hands' clock): object world poses shifted by -delta (G15), the
stored-clock poses and delta (to map robot results back for packing), the RGB camera path, the support-plane gravity
frame (z up, tabletop z = 0, x from the projected camera x) and the intrinsics. GT-free.
usage: ep33_export_for_dexagent.py BUNDLE_EPISODE_DIR CANONICAL_EPISODE_DIR EPISODE OUT_NPZ"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/data1/hansol/mink_S1_20261031/code')
from contact_smooth import DELTA, shift  # noqa: E402


def main():
    bdir, cdir, ep, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    z = np.load(bdir / 'object_poses.npz'); names = [str(n) for n in z['names']]
    F = len(z['world_from_object']); d = DELTA[f'evaluation_{ep}']; d = np.asarray(d, float)[:F] if np.ndim(d) else np.full(F, float(d))
    W_st = z['world_from_object'].astype(float)
    W_rgb = np.stack([shift(W_st[:, k], -d) for k in range(len(names))], 1)
    sp = np.load(cdir / 'support_plane.npz')
    I = json.loads((cdir / 'intrinsics.json').read_text())
    np.savez_compressed(out, names=np.array(names), fps=float(z['fps']), delta=d,
                        world_from_object_rgb=W_rgb, world_from_object_stored=W_st,
                        world_from_rgb=shift(z['world_from_rgb'].astype(float), -d), world_from_rgb_stored=z['world_from_rgb'].astype(float),
                        gravity_from_canonical=sp['gravity_from_canonical'].astype(float),
                        intrinsics=np.array([I['fx'], I['fy'], I['cx'], I['cy'], I['width'], I['height']], float))
    G = sp['gravity_from_canonical']; c0 = (G @ np.r_[W_rgb[0, 0, :3, 3], 1])[:3]
    cam = (G @ z['world_from_rgb'][0])[:3, 3]
    print(json.dumps(dict(names=names, frames=F, delta=[round(float(d.min()), 3), round(float(d.max()), 3)],
                          camera_f0_gravity=np.round(cam, 3).tolist(), first_object_f0_gravity=np.round(c0, 3).tolist())))


if __name__ == '__main__':
    main()
