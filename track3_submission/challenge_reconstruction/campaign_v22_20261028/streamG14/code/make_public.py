"""Apply the hedge rule to the 20 public/held-out clips with the same kind of candidate set that exists on eval (our own trackers:
raw FoundationPose, multimap, pointmap_cotracker, bundle_adjusted where present). Base = r8_base. Writes NEW export folders
canonical/<split>/episode_X/exports/g14_<TAG> (never overwrites). GT is not read here.
usage: make_public.py TAG GP GR MP MR SMOOTH [SOURCES=raw,multimap,pointmap_cotracker,bundle_adjusted]"""
import sys, json, os, shutil, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import hedge as H
W = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical')
TAG = sys.argv[1]; GP, GR, MP, MR = map(float, sys.argv[2:6]); SM = int(sys.argv[6]); MODE = os.environ.get('MODE', 'frame')
SRC = (sys.argv[7] if len(sys.argv) > 7 else 'raw,multimap,pointmap_cotracker,bundle_adjusted').split(',')
clips = sorted(list((W / 'public').glob('episode_*')) + list((W / 'public_heldout').glob('episode_*')))
summ = {}
if os.environ.get('CLIPS'): clips = [c for c in clips if int(c.name[-6:]) in [int(x) for x in os.environ['CLIPS'].split(',')]]
PG = '/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020'
def mesh_ok(p, ours):
    import trimesh
    a = trimesh.load(p, force='mesh', process=False); b = trimesh.load(ours, force='mesh', process=False)
    if len(a.vertices) == len(b.vertices) and np.abs(np.asarray(a.vertices) - np.asarray(b.vertices)).max() < 1e-7: return True
    d = np.abs(np.sort(a.extents) - np.sort(b.extents)).max() * 100
    return d <= 0.3 or (len(a.vertices) == len(b.vertices) and d <= 1.0)
def frolike(ep, names, meshes, F, Wr):
    """old GT-free pipeline (stage-B metric FoundationPose + polish, public_gt_check_20261020) = the fro family on public clips."""
    sys.path[:0] = [PG + '/code', PG + '/code/postprocess']
    import score_public_tracking as S
    T = np.full((F, len(names), 4, 4), np.nan); ms = [None] * len(names)
    for k, n in enumerate(names):
        f = f'{PG}/stereo/episode_{ep:06d}/pp_{n}_metric.npz'; m = f'{PG}/meshes/{n}.obj'
        if not os.path.exists(f) or not mesh_ok(m, meshes[k]): continue
        A = S.stageA_poses(ep, n); X = np.load(f)['polish']; Fa = min(F, len(X), len(A['c2w']))
        T[:Fa, k] = Wr[:Fa] @ np.linalg.inv(A['c2w'][:Fa]) @ X[:Fa]; ms[k] = m
    return T, ms
for root in clips:
    out = root / 'exports' / f'g14_{TAG}'
    if (out / 'object_poses.npz').exists(): print('exists', out); continue
    bd = root / 'exports' / 'r8_base'; b = dict(np.load(bd / 'object_poses.npz')); names = [str(x) for x in b['names']]
    mj = json.loads((bd / 'meshes.json').read_text()); meshes = [mj[n] for n in names]
    F = len(b['world_from_object']); alts = []
    for s in SRC:
        if s == 'frolike':
            T, ms = frolike(int(root.name[-6:]), names, meshes, F, b['world_from_rgb'])
            if any(m is not None for m in ms): alts.append((s, T, ms))
            continue
        p = root / 'exports' / s / 'object_poses.npz'
        if not p.exists(): continue
        a = np.load(p); an = [str(x) for x in a['names']]
        T = np.full((F, len(names), 4, 4), np.nan); ms = [None] * len(names)
        Fa = min(F, len(a['camera_from_object']))
        for k, nm in enumerate(names):
            if nm in an:
                T[:Fa, k] = b['world_from_rgb'][:Fa] @ a['camera_from_object'][:Fa, an.index(nm)]; ms[k] = meshes[k]
        alts.append((s, T, ms))
    ep = int(root.name[-6:])
    Tn, info = H.hedge_clip(b['world_from_object'], names, meshes, alts, GP, GR, MP, MR, SM, ep, MODE)
    out.mkdir(parents=True)
    d = dict(b); d['world_from_object'] = Tn; d['camera_from_object'] = np.linalg.inv(b['world_from_rgb'])[:, None] @ Tn
    np.savez(out / 'object_poses.npz', **d)
    shutil.copy2(bd / 'meshes.json', out / 'meshes.json')
    (out / 'source.txt').write_text(f'G14 hedge of r8_base; mode {MODE} params GP {GP} GR {GR} MP {MP} MR {MR} SMOOTH {SM}; sources {SRC}\n')
    (out / 'g14_info.json').write_text(json.dumps(info, indent=1))
    summ[f'{root.parent.name}/{root.name}'] = info
    print(root.parent.name, root.name, {k: (round(v['move_cm_mean'], 2), round(v['move_deg_mean'], 1), round(v['n_inlier_mean'], 2)) for k, v in info['objects'].items()}, flush=True)
json.dump(summ, open(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG14/out/public_{TAG}_info.json', 'w'), indent=1)
