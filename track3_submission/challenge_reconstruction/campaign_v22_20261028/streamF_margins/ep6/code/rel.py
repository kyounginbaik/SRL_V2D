"""hand(right wrist body)-to-spoon relative pose over time for runs: prints rel translation (cm) & rotation diff vs the first run.
usage: rel.py FRAMES RUN_DIR [RUN_DIR...]"""
import sys, json, numpy as np, mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation as R
fr = [int(x) for x in sys.argv[1].split(',')]; base = None
for d in sys.argv[2:]:
    r = json.loads((Path(d) / 'result.json').read_text()); tr = np.load(r['execution']['trajectory']); m = mujoco.MjModel.from_xml_path(r['execution']['scene']); dd = mujoco.MjData(m)
    b = [i for i in range(m.nbody) if m.body(i).name.startswith('robot_right_') and m.body(i).name.endswith('hand_C_MC')] or [i for i in range(m.nbody) if m.body(i).name.startswith('robot_right_')][:1]
    out = []
    for f in fr:
        dd.qpos[:] = tr['qpos'][f]; mujoco.mj_forward(m, dd)
        Th = np.eye(4); Th[:3, :3] = dd.xmat[b[0]].reshape(3, 3); Th[:3, 3] = dd.xpos[b[0]]
        s = m.body('wooden_spoon').id; Ts = np.eye(4); Ts[:3, :3] = dd.xmat[s].reshape(3, 3); Ts[:3, 3] = dd.xpos[s]
        out.append(np.linalg.inv(Th) @ Ts)
    if base is None: base = out
    print(Path(d).name, m.body(b[0]).name, ' '.join('f%d:%.1fcm/%.0fdg' % (f, np.linalg.norm(o[:3, 3] - q[:3, 3]) * 100, np.degrees(R.from_matrix(o[:3, :3] @ q[:3, :3].T).magnitude())) for f, o, q in zip(fr, out, base)))
