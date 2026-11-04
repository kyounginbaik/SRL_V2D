"""G18 task 4: kit-style scoring of the v28 spoon-episode components against OUR target (target_final_v14_g17) when the
hidden 'truth' spoon is a GT-free variant of our own mesh (placed in a random scan frame), for the submitted spoon mesh (A)
and for candidate fix meshes. Non-spoon objects: reference = our target mesh in its own frame (exact registration).
usage: scoresim.py EP OUT.json NDRAW TRUTH1 [TRUTH2 ...]   (candidates from env G18_CANDS, default A,O,B6,B7)"""
import os, sys, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/code')
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g18lib import *
from g10lib import load_sub, load_target, Scorer
PKG = '/data1/SRL_V2D/challenge_submissions_SRL/track3_v28_20261029/all_20_episodes'
TGT = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v14_g17'
EP, OUT, ND, truths = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4:]
CANDS = os.environ.get('G18_CANDS', 'A,O,B6,B7').split(',')
names, subT, subM = load_sub(PKG, EP)
refT, refM = load_target(TGT, EP, names)
si = names.index('wooden_spoon')
bv, bf = load(SPOON)
# frame of the submitted spoon mesh relative to ours (identity for the 5fcd334c episodes; ep15/16 use an older mesh)
Tsub = np.eye(4)
fr = json.loads((G / 'out/subframe.json').read_text()) if (G / 'out/subframe.json').exists() else {}
if str(EP) in fr: Tsub = np.array(fr[str(EP)]['T_sub_from_ours'])


def to_sub(vf):
    v, f = vf; return v @ Tsub[:3, :3].T + Tsub[:3, 3], f


def H(R, t):
    X = np.eye(4); X[:3, :3] = R; X[:3, 3] = t; return X


class Sc(Scorer):
    def __init__(self, regs, anchors, axes, orders):
        self.names = names; self.regs = regs; self.anchor = anchors; self.axes = axes; self.orders = orders


# exact registrations for the non-spoon objects (our target mesh onto itself, kit seed)
base = Scorer(EP, names, [refM[i] if i != si else subM[i] for i in range(len(names))], refM)
res = dict(ep=EP, names=names, truths={})
r0 = base.summary(subT, refT)[0]
res['proxy_v28_vs_target_subcand'] = r0
print(EP, 'proxy (submitted spoon mesh onto our target mesh, no scan frame):', r0, flush=True)
for tn in truths:
    tv, Dinv = truth_variant(tn, bv, bf)
    res['truths'][tn] = {}
    for cn in CANDS:
        cand = subM[si] if cn == 'A' else to_sub(cand_variant(cn, bv, bf))
        rows = []
        for k in range(ND):
            seed = 1_000_003 * EP + 101 * si + 7919 * k
            Rt, tt = rand_rigid(seed + 13)
            rv = tv[0] @ Rt.T + tt; rmesh = (rv, tv[1])
            anc, _ = M._surface_moments(rv, tv[1])
            ca, rot = M._register_object_body(cand, rmesh, anc, seed)
            regs = list(base.regs); regs[si] = (ca, rot)
            anchors = list(base.anchor); anchors[si] = anc
            rT = list(refT); rT[si] = refT[si] @ np.linalg.inv(H(Rt, tt))
            s = Sc(regs, anchors, base.axes, base.orders).summary(subT, rT)[0]
            E = Rt.T @ rot @ Tsub[:3, :3] if cn == 'A' else Rt.T @ rot @ Tsub[:3, :3]
            s['reg_ang'] = round(angle_deg(E), 2); s['reg_roll'] = round(twist_deg(E, long_axis(bv, bf)[1]), 2)
            rows.append(s)
        mp = np.mean([r['MP'] for r in rows])
        so = [r['objects']['wooden_spoon'] for r in rows]
        res['truths'][tn][cn] = dict(MP_frac=float(mp), spoon_pos_med=float(np.median([x[0] for x in so])), spoon_ori_med=float(np.median([x[1] for x in so])),
                                     spoon_ori_max=float(np.max([x[1] for x in so])), worst=min(rows, key=lambda r: (r['MP'], -max(o[1] for o in r['objects'].values()))), rows=rows)
        print(f"{EP} {tn:26s} {cn:4s} MP {mp:.2f} spoon pos/ori med {res['truths'][tn][cn]['spoon_pos_med']:.2f}/{res['truths'][tn][cn]['spoon_ori_med']:.1f}"
              f" max ori {res['truths'][tn][cn]['spoon_ori_max']:.1f} reg ang {np.median([r['reg_ang'] for r in rows]):.1f} worst {res['truths'][tn][cn]['worst']['objects']}", flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
