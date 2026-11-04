"""GT-free ball-driven roll track: Viterbi over roll bins (relative to final_v6, about the spoon long axis) with unary =
2D gray-ball reprojection px (rollscan) on frames whose detections are spoon balls (best px < PXMAX), transition cost
LAM * |delta roll| (wrapped, deg) + a stay-near-ours prior MU * |roll| (deg). Then the v24/v25 rollouts are scored against
final_v6 with the spoon replaced by the rolled track (kit-faithful).
usage: balltrack.py EP [LAM=0.05] [MU=0.0] [PXMAX=20] [TAG]"""
import sys, json, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG10_mp/code')
from g10lib import *
EP = int(sys.argv[1]); LAM = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05; MU = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
PXMAX = float(sys.argv[4]) if len(sys.argv) > 4 else 20.; TAG = sys.argv[5] if len(sys.argv) > 5 else f'l{LAM}_m{MU}_p{PXMAX:g}'
R = np.load(f'{C}/streamG10_mp/out/rollscan_ep{EP}.npz'); th, ball = R['th'], R['ball']; nf, nt = ball.shape
ax, cen = R['axis'], R['centroid']
U = np.zeros((nf, nt)); inf = np.zeros(nf, bool)
for f in range(nf):
    b = ball[f]
    if np.all(np.isnan(b)) or np.nanmin(b) > PXMAX: continue
    inf[f] = True; U[f] = np.minimum(b, 40.) / 4.0      # px/4: a 4 px difference = 1 unit
thw = np.where(th > 180, th - 360, th)
MUGAP = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0   # pull toward our pose only on frames WITHOUT usable balls
D = np.abs(((th[:, None] - th[None]) + 180) % 360 - 180)
# the static-window offset (ball model vs mesh) is the 'ours' reference for the gap prior
o0 = None
def prior(f):
    return MU * np.abs(thw) + (0 if inf[f] else MUGAP) * np.abs(((th - OFF0) + 180) % 360 - 180)
_b0 = [int(np.nanargmin(ball[f])) for f in range(min(24, nf)) if inf[f]]
OFF0 = float(np.median([thw[j] for j in _b0])) if _b0 else 0.0
cost = U[0] + prior(0); bp = np.zeros((nf, nt), int)
for f in range(1, nf):
    tot = cost[:, None] + LAM * D; bp[f] = np.argmin(tot, 0); cost = tot[bp[f], np.arange(nt)] + U[f] + prior(f)
path = np.zeros(nf, int); path[-1] = int(np.argmin(cost))
for f in range(nf - 1, 0, -1): path[f - 1] = bp[f, path[f]]
roll_raw = thw[path].astype(float)
# the ball model's body frame differs from our mesh's by a constant roll (static frames prefer it); remove it so frame 0 = ours
OFF = float(np.median(roll_raw[:24]))
roll = (roll_raw - OFF + 180) % 360 - 180
FLIPONLY = int(sys.argv[6]) if len(sys.argv) > 6 else 1
if FLIPONLY: roll[np.abs(roll) < 40] = 0.0      # keep our pose where the balls do not ask for a large roll
roll[0] = 0.0
# unwrap-free: roll angle per frame relative to v6
B6 = f'{RR}/selected20_final_v6'
subs = {pk: load_sub(PKG[pk], EP) for pk in ('v24', 'v25')}
names = subs['v24'][0]
refT6, refM, d6 = load_bundle(B6, EP, names); si = names.index('wooden_spoon')
T = body_roll(refT6[si], cen, ax, roll)
out = dict(ep=EP, tag=TAG, offset=OFF, mugap=MUGAP, fliponly=FLIPONLY, lam=LAM, mu=MU, pxmax=PXMAX, informative_frames=int(inf.sum()), flipped_frac=round(float((np.abs(roll) > 90).mean()), 3),
           roll=roll.tolist(), segments=[])
t = 0
while t < nf:
    if abs(roll[t]) > 45:
        a = t
        while t < nf and abs(roll[t]) > 45: t += 1
        out['segments'].append([a, t - 1, float(np.median(roll[a:t]))])
    else: t += 1
for pk in subs:
    refT = list(refT6); refT[si] = T
    s, pos, ori = Scorer(EP, names, subs[pk][2], refM).summary(subs[pk][1], refT)
    out[pk] = s
print(EP, TAG, 'offset', OFF, 'informative', int(inf.sum()), 'flipped', out['flipped_frac'], 'segs', [(a, b, round(m)) for a, b, m in out['segments']])
print('   v24', out['v24'], '\n   v25', out['v25'], flush=True)
json.dump(out, open(f'{C}/streamG10_mp/out/balltrack_ep{EP}_{TAG}.json', 'w'))
np.save(f'{C}/streamG10_mp/out/balltrack_ep{EP}_{TAG}_roll.npy', roll)
