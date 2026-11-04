"""Per-frame roll evidence (rollscan) vs candidate tracks' roll relative to final_v6 (twist about the spoon long axis).
usage: rollana.py EP [COMP=c27]  -> out/rollana_epEP.json + printed table"""
import sys, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import C, RR
from scipy.spatial.transform import Rotation as Rot
EP = int(sys.argv[1]); COMP = sys.argv[2] if len(sys.argv) > 2 else 'c27'
R = np.load(f'{C}/streamG10_mp/out/rollscan_ep{EP}.npz'); th, ball, iou, nb = R['th'], R['ball'], R['iou'], R['nb']; ax = R['axis']
d6 = np.load(f'{RR}/selected20_final_v6/episode_{EP:06d}/object_poses.npz'); names = [str(n) for n in d6['names']]; si = names.index('wooden_spoon')
W6 = d6['world_from_object'][:, si]; nf = len(W6)


def twist(Wa, Wb):
    """roll (deg, signed) and swing (deg) of Wb relative to Wa in Wa's body frame about ax"""
    Rr = np.einsum('fji,fjk->fik', Wa[:, :3, :3], Wb[:, :3, :3]); q = Rot.from_matrix(Rr).as_quat()
    p = (q[:, :3] @ ax)[:, None] * ax[None]; tw = np.concatenate([p, q[:, 3:]], 1); n = np.linalg.norm(tw, axis=1, keepdims=True)
    tw = tw / np.maximum(n, 1e-12); roll = np.degrees(2 * np.arctan2(tw[:, :3] @ ax, tw[:, 3]))
    roll = (roll + 180) % 360 - 180
    sw = Rot.from_quat(q) * Rot.from_quat(tw).inv(); return roll, np.degrees(sw.magnitude())


tracks = {}
for lab, path in [('v7rot', f'{RR}/selected20_final_v7_g9rot')]:
    try:
        d = np.load(f'{path}/episode_{EP:06d}/object_poses.npz'); tracks[lab] = d['world_from_object'][:, si]
    except Exception: pass
for cmp in ('c27', 'c21'):
    try:
        z = np.load(f'{C}/streamG10_mp/out/implied_ep{EP}_{cmp}.npz'); tracks[cmp] = z['rot']
    except Exception: pass
rolls = {k: twist(W6, v) for k, v in tracks.items()}
# per-frame preferred roll
def pref(arr, best_min=True):
    out = np.full(nf, np.nan); conf = np.full(nf, np.nan)
    for f in range(nf):
        a = arr[f]
        if np.all(np.isnan(a)): continue
        j = np.nanargmin(a) if best_min else np.nanargmax(a); out[f] = th[j] if th[j] <= 180 else th[j] - 360
        conf[f] = (a[0] - a[j]) if best_min else (a[j] - a[0])
    return out, conf
bp, bc = pref(ball, True); ip, ic = pref(iou, False)
def at(arr, f, ang):
    j = int(np.round((ang % 360) / (th[1] - th[0]))) % len(th); return arr[f, j]
rows = []
for f in range(nf):
    r = dict(f=f, nb=int(nb[f]), ball0=None if np.isnan(ball[f, 0]) else round(float(ball[f, 0]), 1),
             ball_best=None if np.isnan(bp[f]) else [float(bp[f]), round(float(np.nanmin(ball[f])), 1)],
             iou0=None if np.isnan(iou[f, 0]) else round(float(iou[f, 0]), 3), iou_best=None if np.isnan(ip[f]) else [float(ip[f]), round(float(np.nanmax(iou[f])), 3)])
    for k, (ro, sw) in rolls.items():
        r[f'{k}_roll'] = round(float(ro[f]), 0); r[f'{k}_swing'] = round(float(sw[f]), 0)
        if not np.isnan(ball[f, 0]): r[f'{k}_ball'] = round(float(at(ball, f, ro[f])), 1)
        if not np.isnan(iou[f, 0]): r[f'{k}_iou'] = round(float(at(iou, f, ro[f])), 3)
    rows.append(r)
json.dump(rows, open(f'{C}/streamG10_mp/out/rollana_ep{EP}.json', 'w'))
ks = list(rolls)
print('f nb | ball0 bestroll/px | iou0 bestroll/iou | ' + ' | '.join(f'{k} roll/swing ball iou' for k in ks))
for r in rows:
    print(r['f'], r['nb'], '|', r['ball0'], r['ball_best'], '|', r['iou0'], r['iou_best'], '|',
          ' | '.join(f"{r.get(k+'_roll')}/{r.get(k+'_swing')} {r.get(k+'_ball')} {r.get(k+'_iou')}" for k in ks))
