"""R33: per-frame contact / slip diagnostic of robot rollouts (GT-free): spoon-cup penetration (deepest contact between their geoms),
number of right-hand geoms touching the spoon, left-hand geoms touching the cup, spoon rotation change relative to the right palm and
cup rotation change relative to the left palm (deg/frame), spoon angular acceleration. Per 0.5 s. usage: r33contact.py EVAL_DIR [...]"""
import sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as Rt
K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33'); T = Path('/data1/hansol/retarget33')
sc = json.load(open(K / 'runs/save_K33_cwHGAfix_ep33_HGA_c9/actor/evaluation/result.json'))['execution']['scene']
m = mujoco.MjModel.from_xml_path(sc); d = mujoco.MjData(m)
bs, bc = m.body('wooden_spoon').id, m.body('blue_cup').id; pr, pl = m.body('robot_right_right_hand_C_MC').id, m.body('robot_left_left_hand_C_MC').id
def root(b):
    while m.body_parentid[b] != 0: b = m.body_parentid[b]
    return b
rr, rl = root(pr), root(pl)
for ev in sys.argv[1:]:
    Q = np.load(T / 'runs' / ev / 'eval.npz')['Q']; n = len(Q); pen = np.zeros(n); fs = np.zeros(n, int); fc = np.zeros(n, int); tab = np.zeros(n)
    Rs, Rc, Rp, Rq = [], [], [], []
    for k, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_forward(m, d); gs, gc = set(), set()
        for i in range(d.ncon):
            c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]; pair = {b1, b2}
            if pair == {bs, bc}: pen[k] = min(pen[k], c.dist)
            for ob, hand, S in ((bs, rr, gs), (bc, rl, gc)):
                if ob in pair:
                    o = (pair - {ob}).pop() if len(pair) == 2 else ob
                    if root(o) == hand: S.add(c.geom1 if b1 != ob else c.geom2)
        fs[k], fc[k] = len(gs), len(gc)
        Rs.append(d.xmat[bs].reshape(3, 3).copy()); Rc.append(d.xmat[bc].reshape(3, 3).copy()); Rp.append(d.xmat[pr].reshape(3, 3).copy()); Rq.append(d.xmat[pl].reshape(3, 3).copy())
    Rs, Rc, Rp, Rq = map(lambda a: Rt.from_matrix(np.array(a)), (Rs, Rc, Rp, Rq))
    sl = lambda Ro, Rh: np.r_[0, np.degrees(((Rh[1:].inv() * Ro[1:]) * (Rh[:-1].inv() * Ro[:-1]).inv()).magnitude())]
    ss, cs = sl(Rs, Rp), sl(Rc, Rq); w = (Rs[1:] * Rs[:-1].inv()).as_rotvec(); aa = np.r_[0, 0, np.degrees(np.linalg.norm(np.diff(w, axis=0), axis=1))]
    print(f'\n== {ev}: spoon-cup penetration frames {int((pen < -0.002).sum())} (deepest {100*pen.min():.1f} cm) | spoon slip in right palm >5 deg/frame: {int((ss[55:300] > 5).sum())} frames, mean {ss[55:300].mean():.2f}'
          f' | cup slip in left palm >5 deg/frame: {int((cs[30:320] > 5).sum())} frames, mean {cs[30:320].mean():.2f}')
    print('   t   | pen_cm (min) pen_frames | spoon: fingers slip_deg max | cup: fingers slip_deg max | spoon ang-acc max')
    for a in range(40, n, 10):
        s = slice(a, min(a + 10, n))
        print(f'  {a/20:5.1f} | {100*pen[s].min():6.1f} {int((pen[s] < -0.002).sum()):3d} | {fs[s].mean():5.1f} {ss[s].mean():5.1f} {ss[s].max():5.1f} | {fc[s].mean():5.1f} {cs[s].mean():5.1f} {cs[s].max():5.1f} | {aa[s].max():6.1f}')
