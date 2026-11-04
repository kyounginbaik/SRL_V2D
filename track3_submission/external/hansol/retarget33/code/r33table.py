"""R33: objects vs the measured tabletop (setup frame, tabletop z = 0) in both target worlds: lowest mesh point, cup axis tilt from the
table normal, spoon resting height, per time. Also the static span (f0-f40) where both objects should rest on the table."""
import numpy as np, trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot
exec(open('/data1/hansol/retarget33/code/r33cupspoon.py').read().split('def robot(')[0].split('K = Path(')[0])
def cup_geom(v):
    c = v.mean(0); w, U = np.linalg.eigh(np.cov((v - c).T)); k = 0 if abs(np.log(w[1] / w[0])) > abs(np.log(w[2] / w[1])) else 2
    ax = U[:, k]; h = (v - c) @ ax; r = np.linalg.norm((v - c) - np.outer(h, ax), axis=1); R_ = np.percentile(r, 95); lo, hi = h.min(), h.max(); near = r < 0.5 * R_
    if not near[h < lo + 0.15 * (hi - lo)].mean() > near[h > hi - 0.15 * (hi - lo)].mean(): ax = -ax
    return c, ax          # ax points from bottom to the open end
R = Path('/data1/hansol/retarget33')
for lab in ('R33_HGA7b', 'R33_HGA_r6'):
    W = R / 'worlds' / lab / 'ep33/episode_corrected/episode_000033'; T = R / 'targets' / lab / 'episode_000033'
    z = np.load(W / 'reference_trajectory.npz', allow_pickle=True); S = z['setup_from_source']
    print(f'\n=== {lab}  (setup frame: tabletop z = 0, table centre (13.4, -8.4) cm, radius 22.7 cm)')
    for n in ('blue_cup', 'wooden_spoon'):
        Tw = np.load(T / f'{n}_trajectory.npz')['T_world_obj']; Ts = np.einsum('ij,fjk->fik', S, Tw); v = np.asarray(trimesh.load(T / f'{n}_mesh_metric.obj', force='mesh').vertices, float)
        i = [str(x) for x in z['object_names']].index(n); dp = np.abs(100 * (Ts[:, :3, 3] - z['object_positions'][:, i])).max()
        print(f'  {n}: |setup_from_source*T - reference object_positions| max {dp:.2f} cm')
        if n == 'blue_cup': c, ax = cup_geom(v)
        for f in (0, 10, 20, 30, 40, 60, 80, 120, 200, 280, 330, 347):
            X = v @ Ts[f, :3, :3].T + Ts[f, :3, 3]; s = f'    f{f:3d} {f/20:5.2f}s  centre {np.round(100*Ts[f,:3,3],1).tolist()}  lowest point z {100*X[:,2].min():5.1f} cm'
            if n == 'blue_cup':
                a = Ts[f, :3, :3] @ ax; s += f'  axis tilt from table normal {np.degrees(np.arccos(np.clip(a[2], -1, 1))):5.1f} deg  r from table centre {100*np.hypot(*(Ts[f,:2,3]-[0.1337,-0.0842])):.1f} cm'
            print(s)
