"""[E31 copy: scoring target = $E31_TGT (target_E14_HGA / target_E14_HGB)] E31: verify the cowork2 bundles (untrusted data: np.load allow_pickle=False, json only) against selected20_final_v14_g17.
usage: verify_format.py BUNDLE_A BUNDLE_B REF OUT.json"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
A, B, REF, OUT = map(Path, sys.argv[1:5])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def rel(root): return {str(p.relative_to(root)): p for p in sorted(root.rglob('*')) if p.is_file()}
ref = rel(REF); rep = {}
for name, root in (('HGA', A), ('HGB', B)):
    cur = rel(root); r = dict(missing=[], added=[], differ=[], identical=0, npz={}, meshes_identical=True)
    for k in ref:
        if k not in cur: r['missing'].append(k)
        elif sha(ref[k]) == sha(cur[k]): r['identical'] += 1
        else: r['differ'].append(k)
    r['added'] = [k for k in cur if k not in ref and k.startswith('episode_')]
    r['top_added'] = [k for k in cur if k not in ref and not k.startswith('episode_')][:20]
    for k in ref:
        if '/meshes/' in k and k in cur and sha(ref[k]) != sha(cur[k]): r['meshes_identical'] = False
    for ep in sorted(REF.glob('episode_*')):
        a = np.load(ep / 'object_poses.npz', allow_pickle=False); b = np.load(root / ep.name / 'object_poses.npz', allow_pickle=False)
        d = dict(keys_equal=sorted(a.files) == sorted(b.files), frames=[int(len(a['world_from_object'])), int(len(b['world_from_object']))])
        d['dtype_shape_equal'] = all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape for k in a.files if k in b.files)
        d['changed_keys'] = [k for k in a.files if k in b.files and not np.array_equal(a[k], b[k])]
        if 'names' in a.files: d['names_equal'] = list(map(str, a['names'])) == list(map(str, b['names']))
        W0, W1 = a['world_from_object'], b['world_from_object']; nm = [str(x) for x in a['names']]; obj = {}
        for s, n in enumerate(nm):
            dp = np.linalg.norm(W0[:, s, :3, 3] - W1[:, s, :3, 3], axis=-1) * 100
            Rr = np.einsum('nji,njk->nik', W0[:, s, :3, :3], W1[:, s, :3, :3]); ang = np.degrees(np.arccos(np.clip((np.trace(Rr, axis1=1, axis2=2) - 1) / 2, -1, 1)))
            obj[n] = dict(mean_cm=round(float(dp.mean()), 3), max_cm=round(float(dp.max()), 3), mean_deg=round(float(ang.mean()), 2), max_deg=round(float(ang.max()), 2), f0_cm=round(float(dp[0]), 3), f0_deg=round(float(ang[0]), 2))
        d['delta_vs_ref'] = obj
        for f in ('camera.npz',):
            c0 = np.load(ep / f, allow_pickle=False); c1 = np.load(root / ep.name / f, allow_pickle=False)
            d['camera_identical'] = sorted(c0.files) == sorted(c1.files) and all(np.array_equal(c0[k], c1[k]) for k in c0.files)
        r['npz'][ep.name] = d
    rep[name] = r
# HGA vs HGB
ab = {}
for ep in sorted(A.glob('episode_*')):
    a = np.load(ep / 'object_poses.npz', allow_pickle=False); b = np.load(B / ep.name / 'object_poses.npz', allow_pickle=False)
    ab[ep.name] = all(np.array_equal(a[k], b[k]) for k in a.files) and sha(ep / 'object_poses.npz') == sha(B / ep.name / 'object_poses.npz')
rep['HGA_vs_HGB_object_poses_identical'] = ab
OUT.write_text(json.dumps(rep, indent=1))
for n in ('HGA', 'HGB'):
    r = rep[n]; print(n, 'missing', r['missing'], 'identical', r['identical'], 'differ', r['differ'], 'meshes_identical', r['meshes_identical'])
    for e, d in r['npz'].items(): print(' ', e, d['keys_equal'], d['dtype_shape_equal'], d['frames'], d['changed_keys'], d.get('names_equal'), d['camera_identical'], {k: (v['mean_cm'], v['mean_deg'], v['max_deg']) for k, v in d['delta_vs_ref'].items()})
print('A==B', {k: v for k, v in ab.items()})
