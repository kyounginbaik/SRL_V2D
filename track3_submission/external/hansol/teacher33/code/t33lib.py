"""T33 shared helpers (GT-free): scorer-faithful errors vs $E31_TGT (mlibH = kit MP-SR registration + frame-0 alignment, shipped G18 meshes),
per-frame finger contacts (right hand on the spoon, left hand on the cup; counted as distinct FINGERS thumb/index/middle/ring/pinky/palm),
left/right palm azimuth about the cup axis seen from the camera, and a per-0.5 s table."""
import os, sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
LEG = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017'); sys.path[:0] = [str(LEG / 'code'), '/data1/SRL_V2D/third_party/DexAgent', str(Path(__file__).resolve().parent)]
from tool_library.common import hand as Hh
import mlibH as L
MESH = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep33_H13b1/episode_000033/meshes/episode_000033')
W1 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB_ep3233/ep33/W1/episode_corrected/episode_000033')
CUPAX = np.array([-0.0237963577378219, -0.005046973670649213, 0.9997040869253169])   # shipped cup symmetry axis in its body frame (teacher_ilc_b.py)
FING = ('thumb', 'index', 'middle', 'ring', 'pinky')
_reg = {}
def reg(mesh=MESH):
    if str(mesh) not in _reg: _reg[str(mesh)] = L.register(33, {n: Path(mesh) / f'{n}.obj' for n in L.NAMES[33]})
    return _reg[str(mesh)]
def obj_T(m, Q):
    P = {}
    for n in L.NAMES[33]:
        a = int(m.joint(f'{n}_free').qposadr[0]); T = np.tile(np.eye(4), (len(Q), 1, 1))
        T[:, :3, :3] = R.from_quat(Q[:, a + 3:a + 7], scalar_first=True).as_matrix(); T[:, :3, 3] = Q[:, a:a + 3]; P[n] = T
    return P
def errors(m, Q, mesh=MESH):
    e = L.per_frame(33, obj_T(m, Q), reg(mesh)); return {n: (np.asarray(e[n][0]), np.asarray(e[n][1])) for n in L.NAMES[33]}
def summary(e): return {n: (round(float(p.mean()), 3), round(float(d.mean()), 2)) for n, (p, d) in e.items()}
def finger_map(m, side):
    lab = {}
    for j in range(m.njnt):
        for fi in FING:
            if m.joint(j).name.startswith(f'robot_{side}_{side}_{fi}_'): lab[int(m.jnt_bodyid[j])] = fi
    palm = m.body(f'robot_{side}_{side}_hand_C_MC').id
    def root(b):
        while m.body_parentid[b] != 0: b = m.body_parentid[b]
        return b
    rt = root(palm); out = {}
    for b in range(m.nbody):
        if root(b) != rt: continue
        x = b
        while x not in lab and x != rt: x = m.body_parentid[x]
        out[b] = lab.get(x, 'palm')
    return out
def contacts(m, Q):
    """per frame: set of right-hand fingers touching the spoon, left-hand fingers touching the cup, spoon-cup deepest penetration (m)."""
    d = mujoco.MjData(m); bs, bc = m.body('wooden_spoon').id, m.body('blue_cup').id; fr, fl = finger_map(m, 'right'), finger_map(m, 'left')
    S, C, pen = [], [], np.zeros(len(Q))
    for k, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_forward(m, d); s, c = set(), set()
        for i in range(d.ncon):
            g = d.contact[i]; b1, b2 = int(m.geom_bodyid[g.geom1]), int(m.geom_bodyid[g.geom2])
            if {b1, b2} == {bs, bc}: pen[k] = min(pen[k], g.dist)
            for ob, fm, st in ((bs, fr, s), (bc, fl, c)):
                if ob in (b1, b2):
                    o = b2 if b1 == ob else b1
                    if o in fm: st.add(fm[o])
        S.append(s); C.append(c)
    return S, C, pen
def camera_phys():
    man = json.loads((W1 / 'input_manifest.json').read_text()); SfS = np.asarray(man['setup_from_source'])
    cam = np.load(Path(man['source']) / 'object_poses.npz')['world_from_rgb'][:, :3, 3]; return (SfS[:3, :3] @ cam.T).T + SfS[:3, 3]
def azimuth(m, Q):
    """left/right palm azimuth about the cup axis seen from the camera (deg; -90 = cup image-left side), left palm radial/axial offset (cm)."""
    cam = camera_phys(); d = mujoco.MjData(m); bl, br, bc = (m.body(x).id for x in ('robot_left_left_hand_C_MC', 'robot_right_right_hand_C_MC', 'blue_cup'))
    N = len(Q); out = np.zeros((N, 4))
    for k in range(N):
        d.qpos[:] = Q[k]; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d); c = d.xipos[bc]; ax = d.xmat[bc].reshape(3, 3) @ CUPAX; ax = ax if ax[2] > 0 else -ax
        f = cam[min(k, len(cam) - 1)] - c; f = f - (f @ ax) * ax; f /= np.linalg.norm(f); x = np.cross(ax, f)
        for j, b in enumerate((bl, br)):
            o = d.xpos[b] - c; o = o - (o @ ax) * ax; out[k, j] = np.degrees(np.arctan2(o @ x, o @ f))
        o = d.xpos[bl] - c; out[k, 2] = 100 * np.linalg.norm(o - (o @ ax) * ax); out[k, 3] = 100 * (o @ ax)
    return out
ORD = FING + ('palm',)
def table(m, Q, label=''):
    e = errors(m, Q); S, C, pen = contacts(m, Q); az = azimuth(m, Q); N = len(Q)
    rows = []; lines = [f'== {label} | ' + json.dumps(summary(e)),
        '  t0-t1 s  | cup cm  deg | spoon cm  deg | R fingers on spoon: mean, frames>=3 of 10, union | L fingers on cup: mean, union | L phi  R phi  L rad/ax cm | spoon-cup pen cm']
    for a in range(0, N, 10):
        s = slice(a, min(a + 10, N)); fs = [len(x) for x in S[s]]; fc = [len(x) for x in C[s]]
        us = sorted(set().union(*S[s]), key=ORD.index); uc = sorted(set().union(*C[s]), key=ORD.index)
        r = dict(t0=a / 20, t1=min(a + 10, N) / 20, cup=[float(e['blue_cup'][0][s].mean()), float(e['blue_cup'][1][s].mean())], spoon=[float(e['wooden_spoon'][0][s].mean()), float(e['wooden_spoon'][1][s].mean())],
                 r_fingers_mean=float(np.mean(fs)), r_ge3=int(sum(x >= 3 for x in fs)), r_set=us, l_fingers_mean=float(np.mean(fc)), l_set=uc, l_phi=float(np.median(az[s, 0])), r_phi=float(np.median(az[s, 1])),
                 l_rad=float(np.median(az[s, 2])), l_ax=float(np.median(az[s, 3])), pen_cm=float(100 * pen[s].min()))
        rows.append(r)
        lines.append(f'  {r["t0"]:4.1f}-{r["t1"]:4.1f} | {r["cup"][0]:5.2f} {r["cup"][1]:5.1f} | {r["spoon"][0]:6.2f} {r["spoon"][1]:5.1f} | {r["r_fingers_mean"]:3.1f} {r["r_ge3"]:2d} {",".join(x[:2] for x in us):17s} | {r["l_fingers_mean"]:3.1f} {",".join(x[:2] for x in uc):17s} | {r["l_phi"]:5.0f} {r["r_phi"]:5.0f} {r["l_rad"]:5.1f} {r["l_ax"]:5.1f} | {r["pen_cm"]:5.1f}')
    return dict(summary=summary(e), rows=rows), '\n'.join(lines)
