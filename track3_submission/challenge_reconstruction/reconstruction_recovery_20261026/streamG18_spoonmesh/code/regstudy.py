"""G18 task 2: register candidate spoon meshes onto truth variants with the kit's exact routine
(M._register_object_body, kit seeds of the 10 spoon episodes, random scan frame per seed).
usage: regstudy.py OUT.json CANDS(comma) TRUTH1 [TRUTH2 ...]"""
import sys, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/code')
from g18lib import *
out, cands, truths = sys.argv[1], sys.argv[2].split(','), sys.argv[3:]
bv, bf = load(SPOON)
SLOT = {3: 1, 5: 1, 6: 1, 15: 1, 16: 1, 32: 1, 33: 1, 35: 0, 36: 0, 37: 0}
NS = int(__import__('os').environ.get('G18_NSEED', 2))
res = []
for tn in truths:
    T, Dinv = truth_variant(tn, bv, bf)
    for cn in cands:
        C = cand_variant(cn, bv, bf)
        rows = []
        for ep in EPS:
            for k in range(NS):
                seed = 1_000_003 * ep + 101 * SLOT[ep] + 7919 * k
                r = register(C, T, seed, Dinv=Dinv); r.update(ep=ep, k=k); rows.append(r)
        a = np.array([r['ang'] for r in rows]); ro = np.array([r['roll'] for r in rows]); an = np.array([r['anc_mm'] for r in rows])
        s = dict(truth=tn, cand=cn, n=len(rows), ang_med=float(np.median(a)), ang_max=float(a.max()), roll_med=float(np.median(ro)),
                 roll_absmax=float(np.abs(ro).max()), flip_frac=float((a > 90).mean()), gt10_frac=float((a > 10).mean()),
                 anc_med_mm=float(np.median(an)), anc_max_mm=float(an.max()), truth_faces=len(T[1]), rows=rows)
        print(f"{tn:28s} {cn:4s} ang med {s['ang_med']:6.1f} max {s['ang_max']:6.1f} roll med {s['roll_med']:6.1f} |max| {s['roll_absmax']:6.1f}"
              f" flip {s['flip_frac']:.2f} >10 {s['gt10_frac']:.2f} anc {s['anc_med_mm']:.1f}/{s['anc_max_mm']:.1f} mm", flush=True)
        res.append(s)
json.dump(res, open(out, 'w'))
