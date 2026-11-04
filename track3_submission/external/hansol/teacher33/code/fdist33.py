"""T33: per-finger distance of the RIGHT hand to the spoon (mj_geomDistance, min over the finger's geoms vs the spoon geoms, cm; 0 = touching),
the spoon point closest to each finger in the spoon body frame (x along the handle, cm), and the finger flexion command headroom
(joint upper limit - qpos, rad, summed over that finger's flexion joints). Per 0.5 s median. usage: fdist33.py SCENE|auto ROLLOUT.npz[:LABEL] ..."""
import sys, numpy as np, mujoco
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import t33lib as T
sc = sys.argv[1]
for a in sys.argv[2:]:
    f, _, lab = a.partition(':'); z = np.load(f, allow_pickle=True); Q = z['qpos'] if 'qpos' in z.files else z['Q']
    m = mujoco.MjModel.from_xml_path(str(Path(f).parent / 'scene_physics.xml') if sc == 'auto' else sc); d = mujoco.MjData(m)
    fm = T.finger_map(m, 'right'); bs = m.body('wooden_spoon').id
    sg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bs]
    fg = {fi: [g for g in range(m.ngeom) if fm.get(int(m.geom_bodyid[g])) == fi] for fi in T.FING + ('palm',)}
    fj = {fi: [j for j in range(m.njnt) if m.joint(j).name.startswith(f'robot_right_right_{fi}_') and not m.joint(j).name.endswith('_AA')] for fi in T.FING}
    N = len(Q); D = np.full((N, 6), 9.9); X = np.full((N, 6), np.nan); H = np.zeros((N, 5)); ft = np.zeros(6)
    for k in range(N):
        d.qpos[:] = Q[k]; mujoco.mj_forward(m, d); Rs = d.xmat[bs].reshape(3, 3); ps = d.xpos[bs]
        for i, fi in enumerate(T.FING + ('palm',)):
            best = 0.05
            for g in fg[fi]:
                for h in sg:
                    dd = mujoco.mj_geomDistance(m, d, g, h, 0.05, ft)
                    if dd < best: best = dd; X[k, i] = 100 * (Rs.T @ (ft[3:] - ps))[0]
            D[k, i] = 100 * max(best, 0)
        for i, fi in enumerate(T.FING): H[k, i] = sum(m.jnt_range[j][1] - Q[k, m.jnt_qposadr[j]] for j in fj[fi])
    print(f'== {lab or f}: right finger -> spoon distance cm (median per 0.5 s; 5.0 = >= 5 cm) | spoon-frame x of the closest point cm | flexion headroom rad')
    print('   t   | ' + ' '.join(f'{x[:5]:>6s}' for x in T.FING + ('palm',)) + ' | ' + ' '.join(f'{x[:3]:>5s}' for x in T.FING + ('palm',)) + ' | ' + ' '.join(f'{x[:3]:>5s}' for x in T.FING))
    for a0 in range(40, min(N, 320), 10):
        s = slice(a0, a0 + 10)
        print(f' {a0/20:5.1f} | ' + ' '.join(f'{np.median(D[s, i]):6.2f}' for i in range(6)) + ' | ' + ' '.join(f'{np.nanmedian(X[s, i]) if np.isfinite(X[s, i]).any() else np.nan:5.1f}' for i in range(6))
              + ' | ' + ' '.join(f'{np.median(H[s, i]):5.2f}' for i in range(5)))
    # fingertip (last body of each finger) and palm positions in the spoon body frame (cm), median per 1 s, plus the spoon mesh extent
    def depth(b):
        n_ = 0
        while b != 0: b = m.body_parentid[b]; n_ += 1
        return n_
    tips = {fi: max([b for b, f in fm.items() if f == fi], key=depth) for fi in T.FING}
    tips['palm'] = m.body('robot_right_right_hand_C_MC').id
    P = {k: np.zeros((N, 3)) for k in tips}
    for k in range(N):
        d.qpos[:] = Q[k]; mujoco.mj_kinematics(m, d); Rs = d.xmat[bs].reshape(3, 3); ps = d.xpos[bs]
        for fi, b in tips.items(): P[fi][k] = 100 * Rs.T @ (d.xpos[b] - ps)
    print(f'  fingertip bodies: { {fi: m.body(b).name for fi, b in tips.items()} }')
    print('   t   | ' + ' | '.join(f'{fi[:5]:>5s} x    y    z' for fi in tips))
    for a0 in range(40, min(N, 320), 20):
        s = slice(a0, a0 + 20); print(f' {a0/20:5.1f} | ' + ' | '.join(' '.join(f'{v:5.1f}' for v in np.median(P[fi][s], 0)) for fi in tips))
