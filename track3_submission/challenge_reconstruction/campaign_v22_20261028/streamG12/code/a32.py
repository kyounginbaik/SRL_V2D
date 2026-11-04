"""G12 task A: family of GT-free ep32 spoon roll tracks (ball Viterbi + compose27 twist prior on ball-poor frames) scored
kit-faithfully against the three official ep32 components (v24 B9_ep32_s6f0, v25 H9_ep32_h19f0, v26 H10_ep32_H2f0).
Official: v24 SP fail; v25 SP pass + MP fail; v26 SP pass + MP fail. Also per-episode MPPE (kit keypoints) deltas.
Writes out/a32_cands.json and out/a32_rolls.npz.  usage: a32.py"""
import sys, json, itertools, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import *
from scipy.spatial.transform import Rotation as Rot
G = f'{C}/streamG12'; EP = 32
PKG['v26'] = '/data1/SRL_V2D/challenge_submissions_SRL/track3_v26_20261029/all_20_episodes'
PK = ('v24', 'v25', 'v26')
subs = {pk: load_sub(PKG[pk], EP) for pk in PK}; names = subs['v24'][0]
refT6, refM, d6 = load_bundle(f'{RR}/selected20_final_v6', EP, names); si = names.index('wooden_spoon')
sc = {pk: Scorer(EP, names, subs[pk][2], refM) for pk in PK}
KP = M._MPPE_KEYPOINT_DIRECTIONS * M._MPPE_KEYPOINT_RADIUS_M


def full(pk, refT):
    S = sc[pk]; subT = subs[pk][1]
    nf = min(min(len(t) for t in subT), min(len(t) for t in refT))
    ach = np.stack([S.q(t[:nf]) for t in subT], 1)[:, None]; ref = np.stack([S.q(t[:nf]) for t in refT], 1)[:, None]
    for slot in range(len(names)):
        ca, R = S.regs[slot]
        ach[:, :, slot] = M._convert_object_body(ach[:, :, slot], ca, R)
        ref[:, :, slot, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, slot, 3:]), S.anchor[slot])
    al = M._align_registered_scene(ach, ref, S.axes)
    pos = np.linalg.norm(al[..., :3] - ref[..., :3], axis=-1)[:, 0] * 100
    ori = np.degrees(M._symmetry_orientation_errors(al, ref, S.axes, S.orders))[:, 0]
    mppe = float(np.mean([M._symmetry_point_errors(al[:, :, i], ref[:, :, i], KP, S.axes[i], S.orders[i]) for i in range(len(names))]) * 100)
    pm, om = pos.mean(0), ori.mean(0)
    return dict(obj={n: [round(float(pm[i]), 2), round(float(om[i]), 1)] for i, n in enumerate(names)},
                MP=bool(np.all((pm < 3) & (om < 30))), SP=bool(pm.mean() <= 10 and om.mean() <= np.degrees(0.5)),
                m=round(float(om.mean()), 1), mppe=round(mppe, 3), ori_pf=ori[:, si])


R = np.load(f'{C}/streamG10_mp/out/rollscan_ep32.npz'); th, ball = R['th'], R['ball']; ax, cen = R['axis'], R['centroid']
nf, nt = ball.shape; thw = np.where(th > 180, th - 360, th); OFF = -15.0
inf = np.array([not (np.all(np.isnan(ball[f])) or np.nanmin(ball[f]) > 20) for f in range(nf)])
Ub = np.zeros((nf, nt))
for f in range(nf):
    if inf[f]:
        for j in range(nt):
            jb = int(round(((th[j] + OFF) % 360) / 15)) % nt; Ub[f, j] = min(ball[f, jb], 40.) / 4.
# compose27 twist relative to v6 (body frame, about the spoon long axis)
imp = np.load(f'{C}/streamG10_mp/out/implied_ep32_c27.npz')['rot']
Rrel = np.einsum('fji,fjk->fik', refT6[si][:, :3, :3], imp[:, :3, :3]); q = Rot.from_matrix(Rrel).as_quat()
tw = np.degrees(2 * np.arctan2(q[:, :3] @ ax, q[:, 3])); tw = (tw + 180) % 360 - 180
sw = np.degrees(Rot.from_matrix(Rrel).magnitude())
D = np.abs(((th[:, None] - th[None]) + 180) % 360 - 180)


def viterbi(lam, wc, gaponly, fmax=nf, wb=1.0):
    pri = np.zeros((nf, nt))
    for f in range(nf):
        if f < fmax and (not gaponly or not inf[f]):
            pri[f] = wc * np.abs(((th - tw[f]) + 180) % 360 - 180)
    U = wb * Ub + pri
    cost = U[0].copy(); cost[thw != 0] += 1e3; bp = np.zeros((nf, nt), int)
    for f in range(1, nf):
        tot = cost[:, None] + lam * D; bp[f] = np.argmin(tot, 0); cost = tot[bp[f], np.arange(nt)] + U[f]
    p = np.zeros(nf, int); p[-1] = int(np.argmin(cost))
    for f in range(nf - 1, 0, -1): p[f - 1] = bp[f, p[f]]
    r = thw[p].astype(float)
    un = np.degrees(np.unwrap(np.radians(r))); sm = un.copy()
    for t in range(nf): sm[t] = un[max(0, t - 2):t + 3].mean()
    sm[0] = 0.; return sm


def ballcost(roll):
    """mean ball px on informative frames for a roll track (nearest bin)"""
    j = np.round(((roll % 360)) / 15).astype(int) % nt
    return float(np.mean([Ub[f, j[f]] * 4 for f in range(nf) if inf[f]]))


rows = []; rolls = {}
def run(lab, roll):
    T = body_roll(refT6[si], cen, ax, roll); refT = list(refT6); refT[si] = T
    r = {pk: full(pk, refT) for pk in PK}
    ok = (not r['v24']['SP']) and r['v25']['SP'] and (not r['v25']['MP']) and r['v26']['SP'] and (not r['v26']['MP'])
    row = dict(label=lab, ok=ok, ball_px=round(ballcost(roll), 2), flip_frac=round(float((np.abs((roll + 180) % 360 - 180) > 90).mean()), 3),
               **{pk: {k: v for k, v in r[pk].items() if k != 'ori_pf'} for pk in PK},
               dmppe_25_24=round(r['v25']['mppe'] - r['v24']['mppe'], 3), dmppe_26_25=round(r['v26']['mppe'] - r['v25']['mppe'], 3))
    rows.append(row); rolls[lab] = roll
    print(('OK ' if ok else '-- ') + lab, row['ball_px'], '|', ' | '.join(f"{pk} {row[pk]['obj']['wooden_spoon']} m{row[pk]['m']} S{int(row[pk]['SP'])}M{int(row[pk]['MP'])} e{row[pk]['mppe']}" for pk in PK),
          'dM', row['dmppe_25_24'], row['dmppe_26_25'], flush=True)


run('v6', np.zeros(nf))
run('c27twist_all', tw.copy())
t2 = tw.copy(); t2[188:] = 0; run('c27twist_lt188', t2)
for lam in (0.02, 0.05, 0.1):
    for wc in (0, 0.002, 0.005, 0.01, 0.02, 0.05):
        for gap in (1, 0):
            if wc == 0 and gap == 0: continue
            for fmax in (nf, 188):
                if wc == 0 and fmax != nf: continue
                run(f'vit_l{lam}_c{wc}_g{gap}_f{fmax}', viterbi(lam, wc, gap, fmax))
json.dump(dict(rows=rows, informative=np.where(inf)[0].tolist(), c27_twist=np.round(tw, 1).tolist(), c27_swing=np.round(sw, 1).tolist()),
          open(f'{G}/out/a32_cands.json', 'w'), indent=1)
np.savez(f'{G}/out/a32_rolls.npz', **rolls)
