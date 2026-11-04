"""Per-frame contact summary between sweeper, dust pan, left/right hands and table (replays recorded qpos with mj_forward)."""
import sys, json, numpy as np, mujoco
from pathlib import Path
for d in sys.argv[1:]:
    r = json.loads((Path(d) / 'result.json').read_text()) if (Path(d) / 'result.json').exists() else None
    scene = Path(d) / 'execution/scene_physics.xml'; m = mujoco.MjModel.from_xml_path(str(scene)); dd = mujoco.MjData(m)
    z = np.load(Path(d) / 'execution/trajectory.npz'); Q = z['qpos']
    def cls(g):
        n = m.body(m.geom_bodyid[g]).name
        for k in ('mini_sweeper', 'mini_dust_pan', 'robot_left', 'robot_right'):
            if n.startswith(k): return k
        return 'table' if 'table' in (m.geom(g).name or '') else 'other'
    rows = []
    for k in range(0, len(Q), 5):
        dd.qpos[:] = Q[k]; mujoco.mj_forward(m, dd); s = {}
        for i in range(dd.ncon):
            c = dd.contact[i]; a, b = sorted((cls(c.geom1), cls(c.geom2))); s[a + '-' + b] = s.get(a + '-' + b, 0) + 1
        rows.append((k / 20, {x: v for x, v in s.items() if 'mini' in x}))
    print(Path(d).name); [print(' ', t, s) for t, s in rows if 5.0 <= t <= 9.0]
