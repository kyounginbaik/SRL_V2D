"""Score a finished run on our writing rule. Usage: score_any.py spider|daid|rl PATH [world]. Appends to write_results.jsonl and saves the pen path (n,7) in our scene frame at the reference clock."""
import sys, json, re, numpy as np, mujoco, wmetric as W
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
kind, path = sys.argv[1:3]; RF = W.RF; H = W.H; n = RF['n']; os_ = lambda q: q
if kind == 'spider':
    m = mujoco.MjModel.from_xml_path(str(W.P / 'scene.xml')); r = np.load(path); q = r['qpos'].reshape(-1, r['qpos'].shape[-1]); f = int(round(.02 / float(r['time'].reshape(-1)[1] - r['time'].reshape(-1)[0]))); pen = q[::f][:, -7:]; label = 'spider'
elif kind == 'daid':
    O = Path(path); c = (O / 'config.yaml').read_text(); g = lambda k: float(re.search(rf'^{k}: *([-\d.e]+)', c, re.M).group(1)); wu = int(g('warmup_steps')); rs = int(round(g('ref_dt') / g('sim_dt')))
    m = mujoco.MjModel.from_xml_path(str(O / 'scene.xml')); a = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_object_joint')]; r = np.load(O / 'trajectory_mjwp.npz'); q = r['qpos']
    if q.ndim == 3:
        if 'sim_step' in r.files and len(r['sim_step']) == q.shape[0]: q = q[np.argsort(r['sim_step'].ravel())]
        q = q.reshape(-1, m.nq)
    post = q[wu:][:, a:a + 7]; idx = np.arange(0, len(post), rs); pd = post[idx]; M = json.load(open(H / 'board_daid.json')); A = np.array(M['A']); t = np.array(M['t'])
    po = (pd[:, :3] - t) @ A; qo = R.from_matrix(A.T) * R.from_quat(pd[:, 3:][:, [1, 2, 3, 0]])
    # Their frame j is mocap 2109 + 4j. Our frame i is mocap 2109 + (i + shift) * 1364/567.
    mj = 4. * np.arange(len(pd)); mi = (np.arange(n) + RF['shift']) * 1364. / 567.; mi = mi[mi <= mj[-1]]; s = Slerp(mj, qo); pen = np.c_[[np.interp(mi, mj, po[:, x]) for x in range(3)]].T; pen = np.c_[pen, s(mi).as_quat()[:, [3, 0, 1, 2]]]; label = 'daid'
elif kind == 'rl':
    # Policy rolls at 20 Hz: recorded row j is time (j + 1) / 20. Our frame i is time (i + shift) / 50 * (1365 / 120) / (567 * .02).
    d = np.load(path); o = d['o']; w = int(sys.argv[3]) if len(sys.argv) > 3 else 0; bo = o[:, w, 0]; tp = (np.arange(len(bo)) + 1) / 20.; ti = (np.arange(n) + RF['shift']) * .02; ti = ti[ti <= tp[-1]]; rr = R.from_quat(bo[:, 3:7]); pen = np.c_[np.c_[[np.interp(ti, tp, bo[:, x]) for x in range(3)]].T, Slerp(tp, rr)(np.clip(ti, tp[0], tp[-1])).as_quat()[:, [3, 0, 1, 2]]]; label = f'rl {Path(path).parent.parent.name} {Path(path).parent.name} {Path(path).stem} w{w}'
s = W.score(pen, label); s['recorded'] = len(pen); print(json.dumps(s)); open(H / 'write_results.jsonl', 'a').write(json.dumps(s) + '\n'); np.save(H / ('pen_' + re.sub(r'\W+', '_', label) + '.npy'), pen)
