"""SCORER-SIDE ONLY (public scan used exactly as the kit scorer does; nothing from here enters a target or submission).
Kit registration of our ep20 meshes onto the public scan for the official ep20 seeds vs 40 other seeds: is the ep20-seed registration in
the majority basin? A wrong basin would rotate the whole aligned scene (anchor = pitcher frame 0) and is invisible to scan-in-our-frame targets."""
import numpy as np, json, sys
import kit; M = kit.M; K = kit.K
from scipy.spatial.transform import Rotation as Rot
P = '/data1/SRL_V2D/challenge_submissions_SRL/track3_v26_20261029/all_20_episodes/meshes/episode_000020'
out = {}
for slot, n in enumerate(['blue_cup', 'water_pitcher']):
    v, f, anc, ax, o = kit.ref_mesh(K.gt_mesh_path(n), n)
    cand = kit.sub_mesh(f'{P}/{n}.obj')
    seeds = [1_000_003 * 20 + 101 * slot] + [1_000_003 * e + 101 * s for e in (21, 22) for s in (0, 1)] + list(range(1, 33))
    res = []
    for sd in seeds:
        ca, rot = M._register_object_body(cand, (v, f), anc, sd); res.append((sd, ca, rot))
    R0 = res[0][2]; a0 = res[0][1]
    angs = [np.degrees(Rot.from_matrix(r @ R0.T).magnitude()) for _, _, r in res]
    if o == 360:  # symmetric: compare axis only
        angs = [np.degrees(np.arccos(np.clip(abs((r.T @ ax) @ (R0.T @ ax)), -1, 1))) for _, _, r in res]
    sh = [np.linalg.norm(c - a0) * 100 for _, c, _ in res]
    print(n, 'ep20 seed vs others: angle deg', np.round(angs[1:5], 2), 'others p50/p90/max', np.round(np.percentile(angs[5:], [50, 90, 100]), 2),
          ' anchor shift cm p50/max', np.round([np.median(sh[1:]), max(sh[1:])], 2))
    out[n] = dict(angles=angs, shifts=sh)
json.dump(out, open('../scorer_only/s1_regseed.json', 'w'))
