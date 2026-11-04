"""R33: per frame 9.5-12.5 s, spoon bowl centre in the cup frame for the target (r6) and robot runs (R2 on 7b, R8 on r6): axial height
above the cup centroid (rim = cup open end), radial distance, and min distance spoon-cup geoms (robot). usage: r33approach_diag.py"""
import sys
import numpy as np
from pathlib import Path
sys.argv = ['x']; exec(open('/data1/hansol/retarget33/code/r33cupspoon.py').read().split('runs = [')[0])
T = Path('/data1/hansol/retarget33')
(gs, vs), = geom_verts('wooden_spoon')[:1]; (gc, vc), = geom_verts('blue_cup')[:1]
out = {}
for lab, ev, tg in [('R2', 'eval_R2', 'R33_HGA7b'), ('R8', 'eval_R8', 'R33_HGA_r6')]:
    Q = np.load(T / 'runs' / ev / 'eval.npz')['Q']; (pr, rr, hr, ir), cgr = robot(Q); (pt, rt, ht, it), cgt = target(T / 'targets' / tg, len(Q))
    dist = []
    for q in Q[190:252]:
        d.qpos[:] = q; mujoco.mj_forward(m, d); dist.append(mujoco.mj_geomDistance(m, d, gs, gc, 0.2, None))
    out[lab] = (rr, hr, ir, rt, ht, it, cgr, cgt, np.array(dist))
print('frame  t   | target(r6): h-rim r in | R8 robot: h-rim r in dist | target(7b): h-rim r in | R2 robot: h-rim r in dist   (cm)')
for f in range(190, 252, 2):
    a = out['R8']; b = out['R2']; k = f - 190
    print(f"{f:4d} {f/20:5.2f} | {100*(a[4][f]-a[7]['rim']):6.1f} {100*a[3][f]:5.1f} {int(a[5][f])} | {100*(a[1][f]-a[6]['rim']):6.1f} {100*a[0][f]:5.1f} {int(a[2][f])} {100*a[8][k]:5.1f} |"
          f" {100*(b[4][f]-b[7]['rim']):6.1f} {100*b[3][f]:5.1f} {int(b[5][f])} | {100*(b[1][f]-b[6]['rim']):6.1f} {100*b[0][f]:5.1f} {int(b[2][f])} {100*b[8][k]:5.1f}")
