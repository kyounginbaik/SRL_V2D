"""per-frame spoon-pot penetration when the objects sit exactly at the target (qt) vs in the actor rollout."""
import sys, json, numpy as np, mujoco
from pathlib import Path
actor = Path(sys.argv[1]); ev = json.loads((actor / 'evaluation/result.json').read_text())
m = mujoco.MjModel.from_xml_path(ev['execution']['scene']); d = mujoco.MjData(m)
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code', '/data1/SRL_V2D/third_party/DexAgent']
from tool_library.common import tracking as TR
qt = TR.load_reference(Path(ev['input_dir']) / 'tracking_reference.npz', m)['qpos']; Q = np.load(actor / 'evaluation/execution/trajectory.npz')['qpos']
body = {n: m.body(n).id for n in ('white_pot', 'wooden_spoon')} if True else None
def bid_of(g):
    b = m.geom_bodyid[g]
    while m.body_parentid[b] != 0: b = m.body_parentid[b]
    return b
sp = {n: m.joint(n + '_free').qposadr[0] for n in ('white_pot', 'wooden_spoon')}
rootb = {n: m.jnt_bodyid[m.joint(n + '_free').id] for n in sp}
def pen(QQ, objects_only):
    out = []
    for k in range(len(QQ)):
        if objects_only:
            d.qpos[:] = Q[k]
            for n, a in sp.items(): d.qpos[a:a + 7] = qt[k, a:a + 7]
        else: d.qpos[:] = QQ[k]
        mujoco.mj_forward(m, d); worst = 0.0; hand = 0.0
        for c in d.contact[:d.ncon]:
            b1, b2 = bid_of(c.geom1), bid_of(c.geom2); s = {b1, b2}
            if s == {rootb['white_pot'], rootb['wooden_spoon']}: worst = min(worst, c.dist)
        out.append(-worst * 100)
    return np.array(out)
pt = pen(qt, True); pa = pen(Q, False)
print('target objects (hands from rollout): spoon-pot penetration cm  mean %.3f p95 %.3f max %.3f  frames>0.3cm %d / %d' % (pt.mean(), np.percentile(pt, 95), pt.max(), (pt > 0.3).sum(), len(pt)))
print('rollout: mean %.3f p95 %.3f max %.3f' % (pa.mean(), np.percentile(pa, 95), pa.max()))
print('per 0.5 s target max pen:', [round(float(pt[i:i + 10].max()), 2) for i in range(0, len(pt), 10)])
