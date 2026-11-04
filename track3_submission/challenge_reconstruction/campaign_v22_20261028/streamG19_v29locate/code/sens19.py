"""G19 task 3 (GT-free): kit-lever tolerance of the v29 rollout. Truth = our target_final_v14_g17 (our meshes) with ONE object's
whole track given a constant body-frame rotation (about its surface centroid, about each principal axis) or a constant world
translation; the smallest perturbation that makes v29 fail MP-SR (kit-faithful: registration + anchor/slot-0 f0 alignment)
tells how much hidden reconstruction error of that kind the episode tolerates. usage: sens19.py EP -> out/sens/epN.json"""
import sys, os, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG13/code')
import tracks as T, kit, trimesh
from scipy.spatial.transform import Rotation as Rot
T.PKG['v29'] = '/data1/SRL_V2D/challenge_submissions_SRL/track3_v29_20261030/all_20_episodes'
S = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate'
TG = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17'
ep = int(sys.argv[1]); ach = T.package('v29', ep); ref = T.target(None, ep, f'{TG}/episode_{ep:06d}')
names = [r[0] for r in ref]


def score(rf):
    p, o = kit.errors(ach, rf, ep); pm, om = p.mean(0), o.mean(0)
    return bool(np.all((pm < 3) & (om < 30))), [[round(float(a), 2), round(float(b), 1)] for a, b in zip(pm, om)]


base = score(ref); out = dict(ep=ep, names=names, base=base, rot={}, trans={})
print(ep, names, 'base', base, flush=True)
DEG = [2.5, 5, 7.5, 10, 12.5, 15, 20, 25, 30, 40]; CM = [0.5, 1, 1.5, 2, 2.5, 3, 4]
for k, (n, Tw, mp) in enumerate(ref):
    m = trimesh.load(mp, force='mesh'); c = m.centroid; w, U = np.linalg.eigh(np.cov((np.asarray(m.vertices) - c).T))
    for ai in range(3):
        for sgn in (1, -1):
            hit = None; det = None
            for d in DEG:
                Rb = Rot.from_rotvec(U[:, ai] * np.radians(d * sgn)).as_matrix(); C = np.eye(4); C[:3, :3] = Rb; C[:3, 3] = c - Rb @ c
                rf = list(ref); rf[k] = (n, Tw @ C, mp); ok, e = score(rf)
                if not ok: hit, det = d, e; break
            out['rot'][f'{n}/pc{ai}{"+" if sgn > 0 else "-"}'] = dict(fail_at_deg=hit, errors=det)
            print(n, 'pc', ai, sgn, 'fail at', hit, det, flush=True)
    if k == 0 and len(ref) == 1: continue
    for ax in range(3):
        for sgn in (1, -1):
            hit = None; det = None
            for d in CM:
                C = np.eye(4); C[ax, 3] = sgn * d / 100; rf = list(ref); rf[k] = (n, C @ Tw, mp); ok, e = score(rf)
                if not ok: hit, det = d, e; break
            out['trans'][f'{n}/w{"xyz"[ax]}{"+" if sgn > 0 else "-"}'] = dict(fail_at_cm=hit, errors=det)
os.makedirs(f'{S}/out/sens', exist_ok=True); json.dump(out, open(f'{S}/out/sens/ep{ep}.json', 'w'), indent=1)
