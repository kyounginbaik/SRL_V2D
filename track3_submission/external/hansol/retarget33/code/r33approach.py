"""R33: approach guide target (retarget guide only; scoring stays on the source target). Around the 10 s cup entry, the spoon bowl is
pulled onto the cup axis early (radial -> 0) and held MARGIN above the rim until it is on-axis, so the robot enters along the axis instead
of catching the rim from the side. Spoon rotation unchanged; the spoon is translated. usage: r33approach.py SRC_TARGET OUT_TARGET MARGIN_CM LEAD_FRAMES"""
import shutil
import sys
import numpy as np
from pathlib import Path
src, dst, margin, lead = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]) / 100, int(sys.argv[4])
sys.argv = ['x']; exec(open('/data1/hansol/retarget33/code/r33cupspoon.py').read().split('def robot(')[0])
if dst.exists(): shutil.rmtree(dst)
shutil.copytree(src, dst); e = dst / 'episode_000033'
vs = np.asarray(trimesh.load(e / 'wooden_spoon_mesh_metric.obj', force='mesh').vertices, float); vc = np.asarray(trimesh.load(e / 'blue_cup_mesh_metric.obj', force='mesh').vertices, float)
bl = spoon_bowl_local(vs); cg = cup_geom(vc)
Ts = np.load(e / 'wooden_spoon_trajectory.npz')['T_world_obj']; Tc = np.load(e / 'blue_cup_trajectory.npz')['T_world_obj']; n = min(len(Ts), len(Tc))
def ramp(f, a, b, c, d_):   # 0 before a, 1 on [b, c], 0 after d_
    return float(np.clip(min((f - a) / max(b - a, 1), (d_ - f) / max(d_ - c, 1)), 0, 1))
L = lead; out = Ts.copy()
for f in range(185, 215):
    wr, wh = ramp(f, 197 - L, 200 - L, 206, 210), ramp(f, 196 - L, 198 - L, 204, 209)
    if wr == 0 and wh == 0: continue
    Rc, tc = Tc[f, :3, :3], Tc[f, :3, 3]; bw = Ts[f, :3, 3] + Ts[f, :3, :3] @ bl; p = Rc.T @ (bw - tc) - cg['c']
    h = p @ cg['ax']; rv = p - h * cg['ax']; h2 = h + wh * max(0.0, cg['rim'] + margin - h); p2 = h2 * cg['ax'] + (1 - wr) * rv
    out[f, :3, 3] = Ts[f, :3, 3] + Rc @ (p2 - p)
    print(f'f{f} {f/20:.2f}s  w_r {wr:.2f} w_h {wh:.2f}  h-rim {100*(h-cg["rim"]):5.1f} -> {100*(h2-cg["rim"]):5.1f}  r {100*np.linalg.norm(rv):5.1f} -> {100*(1-wr)*np.linalg.norm(rv):5.1f}  move {100*np.linalg.norm(p2-p):4.1f} cm')
np.savez_compressed(e / 'wooden_spoon_trajectory.npz', T_world_obj=out); print('wrote', dst)
