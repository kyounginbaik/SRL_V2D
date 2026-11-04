"""R33: which right-hand fingers touch the spoon, and does the left hand touch the spoon / right hand, per 0.5 s (GT-free).
Finger = the hand link name of the touching geom's body (thumb/index/middle/ring/pinky/palm). usage: r33fingers.py EVAL_DIR"""
import sys, json, re, numpy as np, mujoco
from pathlib import Path
from collections import Counter
K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33'); T = Path('/data1/hansol/retarget33')
sc = json.load(open(K / 'runs/save_K33_cwHGAfix_ep33_HGA_c9/actor/evaluation/result.json'))['execution']['scene']
m = mujoco.MjModel.from_xml_path(sc); d = mujoco.MjData(m); bs, bc = m.body('wooden_spoon').id, m.body('blue_cup').id
def root(b):
    while m.body_parentid[b] != 0: b = m.body_parentid[b]
    return b
rr, rl = root(m.body('robot_right_right_hand_C_MC').id), root(m.body('robot_left_left_hand_C_MC').id)
def finger(b):
    n = m.body(b).name.lower()
    for k in ('thumb', 'index', 'middle', 'ring', 'pinky', 'little'):
        if k in n: return k
    return 'palm' if ('mc' in n or 'palm' in n or 'base' in n) else re.sub(r'robot_(left|right)_(left|right)_', '', n)
Q = np.load(T / 'runs' / sys.argv[1] / 'eval.npz')['Q']; n = len(Q); rows = []
for k, q in enumerate(Q):
    d.qpos[:] = q; mujoco.mj_forward(m, d); fs = set(); ls = 0; lr = 0
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        for a, b in ((b1, b2), (b2, b1)):
            if a == bs and root(b) == rr: fs.add(finger(b))
            if a == bs and root(b) == rl: ls += 1
        if {root(b1), root(b2)} == {rr, rl}: lr += 1
    rows.append((fs, ls, lr))
print(f'== {sys.argv[1]}: right-hand parts touching the spoon, left-hand contacts with the spoon and with the right hand')
print('   t   | spoon touched by (share of frames)                          | L-spoon frames | L-R hand frames')
for a in range(40, n, 10):
    s = rows[a:a + 10]; c = Counter(f for fs, _, _ in s for f in fs); nf = np.mean([len(fs) for fs, _, _ in s])
    print(f'  {a/20:5.1f} | n={nf:3.1f} ' + ' '.join(f'{k}:{v/len(s):.1f}' for k, v in sorted(c.items(), key=lambda x: -x[1])).ljust(52) + f' | {sum(1 for _, x, _ in s if x):2d}/{len(s)} | {sum(1 for _, _, x in s if x):2d}/{len(s)}')
