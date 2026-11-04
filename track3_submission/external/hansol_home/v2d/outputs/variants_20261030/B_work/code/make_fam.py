"""Variant B: build per-family public/held-out exports in our own mirror (B_work/canonical/<split>/episode_X/exports/fam_<family>).
Each object = r8_base world_from_rgb @ family camera_from_object (same camera path as the base); frames the family lacks or that
are non-finite fall back to r8_base and are counted. Meshes = r8_base meshes.json (our meshes, unchanged). GT is not read here.
usage: make_fam.py [families...]"""
import sys, os, json, shutil, numpy as np
from pathlib import Path
SRC = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical')
DST = Path('/home/hansol/v2d/outputs/variants_20261030/B_work/canonical')
PG = '/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020'
FAMS = sys.argv[1:] or ['raw', 'multimap', 'pointmap_cotracker', 'bundle_adjusted', 'frolike']

def mesh_ok(p, ours):
    import trimesh
    a = trimesh.load(p, force='mesh', process=False); b = trimesh.load(ours, force='mesh', process=False)
    if len(a.vertices) == len(b.vertices) and np.abs(np.asarray(a.vertices) - np.asarray(b.vertices)).max() < 1e-7: return True
    d = np.abs(np.sort(a.extents) - np.sort(b.extents)).max() * 100
    return d <= 0.3 or (len(a.vertices) == len(b.vertices) and d <= 1.0)

def family_cam(fam, root, ep, names, meshes, F):
    """camera_from_object per object name (F,4,4) or None."""
    out = {}
    if fam == 'frolike':
        sys.path[:0] = [PG + '/code', PG + '/code/postprocess']
        import score_public_tracking as S
        for k, n in enumerate(names):
            f = f'{PG}/stereo/episode_{ep:06d}/pp_{n}_metric.npz'; m = f'{PG}/meshes/{n}.obj'
            if not os.path.exists(f) or not os.path.exists(m) or not mesh_ok(m, meshes[k]): continue
            A = S.stageA_poses(ep, n); X = np.load(f)['polish']; Fa = min(F, len(X), len(A['c2w']))
            C = np.full((F, 4, 4), np.nan); C[:Fa] = np.linalg.inv(A['c2w'][:Fa]) @ X[:Fa]; out[n] = C
        return out
    p = root / 'exports' / fam / 'object_poses.npz'
    if not p.exists(): return out
    a = np.load(p); an = [str(x) for x in a['names']]
    for n in names:
        if n in an:
            C = np.full((F, 4, 4), np.nan); Ca = a['camera_from_object'][:, an.index(n)]; Fa = min(F, len(Ca)); C[:Fa] = Ca[:Fa]; out[n] = C
    return out

report = {}
for split in ['public', 'public_heldout']:
    for root in sorted((SRC / split).glob('episode_*')):
        ep = int(root.name[-6:]); bd = root / 'exports' / 'r8_base'
        b = dict(np.load(bd / 'object_poses.npz')); names = [str(x) for x in b['names']]
        mj = json.loads((bd / 'meshes.json').read_text()); meshes = [mj[n] for n in names]; F = len(b['world_from_object'])
        for fam in FAMS:
            Cd = family_cam(fam, root, ep, names, meshes, F)
            if not Cd: continue
            T = b['world_from_object'].copy(); cov = {}
            for k, n in enumerate(names):
                if n not in Cd: cov[n] = 0.0; continue
                Wf = b['world_from_rgb'] @ Cd[n]; ok = np.isfinite(Wf).all((1, 2))
                T[ok, k] = Wf[ok]; cov[n] = float(ok.mean())
            out = DST / split / root.name / 'exports' / f'fam_{fam}'
            out.mkdir(parents=True, exist_ok=True)
            d = dict(b); d['world_from_object'] = T; d['camera_from_object'] = np.linalg.inv(b['world_from_rgb'])[:, None] @ T
            np.savez(out / 'object_poses.npz', **d); shutil.copy2(bd / 'meshes.json', out / 'meshes.json')
            (out / 'source.json').write_text(json.dumps(dict(family=fam, base='r8_base', coverage=cov), indent=1))
            report[f'{split}/{ep}/{fam}'] = cov
            print(split, ep, fam, cov, flush=True)
        # the base itself, for exact comparison in the same mirror
        out = DST / split / root.name / 'exports' / 'fam_base'
        if not out.exists():
            out.mkdir(parents=True); [shutil.copy2(bd / f, out / f) for f in ['object_poses.npz', 'meshes.json']]
            if (bd / 'public_score.json').exists(): shutil.copy2(bd / 'public_score.json', out / 'public_score.json')
Path(DST.parent / 'fam_coverage.json').write_text(json.dumps(report, indent=1))
