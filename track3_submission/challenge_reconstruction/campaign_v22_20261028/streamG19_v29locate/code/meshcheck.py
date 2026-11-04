"""For candidate components: md5 of each object mesh vs the v29 submitted mesh and vs the public GT scan mesh (geometry-level:
vertex count + sorted extents), to enforce 'no public scan/GT mesh in any submitted mesh'. usage: meshcheck.py EP [EP...]"""
import sys, os, json, hashlib, glob, numpy as np, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
CD = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates'; OV = '/home/exaFLOPs/v22_overflow_20261028/common_v2_candidates'
V29 = '/data1/SRL_V2D/challenge_submissions_SRL/track3_v29_20261030/all_20_episodes/meshes'
md5 = lambda p: hashlib.md5(open(p, 'rb').read()).hexdigest()[:8]
def sig(p):
    m = trimesh.load(p, force='mesh'); e = np.sort(m.extents); return len(m.vertices), np.round(e, 4)
out = {}
for ep in map(int, sys.argv[1:]):
    n = f'episode_{ep:06d}'
    for d in sorted(glob.glob(f'{CD}/*/{n}') + glob.glob(f'{OV}/*/{n}')):
        tag = d.split('/')[-2]; r = {}
        for mp in sorted(glob.glob(f'{d}/meshes/{n}/*.obj')):
            name = os.path.basename(mp)[:-4]; h = md5(mp); v = f'{V29}/{n}/{name}.obj'
            try: gp = K.gt_mesh_path(name)
            except Exception: gp = None
            gt = None
            if gp and os.path.exists(gp):
                nv, ex = sig(mp); gnv, gex = sig(gp); gt = bool(nv == gnv or np.abs(ex - gex).max() < 0.002) or md5(gp) == h
            r[name] = dict(md5=h, same_as_v29=(h == md5(v)) if os.path.exists(v) else None, looks_like_gt=gt)
        out[f'{ep}/{tag}'] = r
        print(ep, tag, {k: (x['md5'], 'v29' if x['same_as_v29'] else '-', 'GT!' if x['looks_like_gt'] else '') for k, x in r.items()}, flush=True)
json.dump(out, open(f'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate/out/meshcheck_{"_".join(sys.argv[1:])}.json', 'w'), indent=1)
