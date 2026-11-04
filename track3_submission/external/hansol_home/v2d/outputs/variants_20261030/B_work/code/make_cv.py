"""Variant B 2-fold check: apply the B2 rule derived from one split (cv_rules.json) to the OTHER split -> exports/B2cv.
Unchanged clips copy r8_base's score. GT is not read here."""
import sys, os, json, shutil, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__)); import variants as V
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical')
W = Path('/home/hansol/v2d/outputs/variants_20261030/B_work'); cv = json.load(open(W / 'cv_rules.json')); lines = []
for split, fold in [('public', 'heldout_only'), ('public_heldout', 'public_only')]:
    rules = (cv[fold]['B1'], cv[fold]['B2_good'])
    for root in sorted((SRC / split).glob('episode_*')):
        bd = root / 'exports' / 'r8_base'; b = dict(np.load(bd / 'object_poses.npz')); mj = json.loads((bd / 'meshes.json').read_text())
        T, info = V.apply(b, str(root / 'exports'), mj, rules, 'B2')
        out = W / 'canonical' / split / root.name / 'exports' / 'B2cv'
        if out.exists(): shutil.rmtree(out)
        out.mkdir(parents=True)
        same_as_b2 = (W / 'canonical' / split / root.name / 'exports' / 'B2' / 'object_poses.npz')
        if np.abs(T - b['world_from_object']).max() <= 1e-9:
            for f in ['object_poses.npz', 'meshes.json', 'public_score.json']: shutil.copy2(bd / f, out / f)
            continue
        d = dict(b); d['world_from_object'] = T; d['camera_from_object'] = np.linalg.inv(b['world_from_rgb'])[:, None] @ T
        if same_as_b2.exists() and np.abs(np.load(same_as_b2)['world_from_object'] - T).max() <= 1e-12 and (same_as_b2.parent / 'public_score.json').exists():
            for f in ['object_poses.npz', 'meshes.json', 'public_score.json']: shutil.copy2(same_as_b2.parent / f, out / f)
            print(split, root.name, 'identical to B2'); continue
        np.savez(out / 'object_poses.npz', **d); shutil.copy2(bd / 'meshes.json', out / 'meshes.json')
        (out / 'source.json').write_text(json.dumps(dict(fold=fold, info=info), indent=1)); lines.append(str(out))
        print(split, root.name, fold, {k: v.get('source') for k, v in info.items()}, flush=True)
(W / 'lists' / 'cv.txt').write_text('\n'.join(lines) + '\n'); print(len(lines), 'to score')
