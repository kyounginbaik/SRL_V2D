"""Spoon-in-cup measure: depth of the spoon's bowl end below the cup rim along the cup axis (positive = inside),
and radial offset from the cup axis, for the reference track and for physical rollouts (trajectory.npz or results parquet).
usage: spoon_in_cup.py EP label:path [label:path ...]"""
import sys, numpy as np, trimesh, pandas as pd
from scipy.spatial.transform import Rotation as R
ep = int(sys.argv[1]); P = '/data1/SRL_V2D/challenge_submissions/track3_v9_20261021/all_20_episodes'
src = f'/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_{ep:06d}'
z = np.load(src + '/object_poses.npz'); names = [str(x) for x in z['names']]; T = z['world_from_object']
mesh = {n: np.asarray(trimesh.load(f'{P}/meshes/episode_{ep:06d}/{n}.obj', force='mesh').vertices) for n in names}
cv = mesh['blue_cup']; c = cv - cv.mean(0); w, U = np.linalg.eigh(c.T @ c)
g = [abs(w[k]-w[(k+1)%3]) + abs(w[k]-w[(k+2)%3]) - abs(w[(k+1)%3]-w[(k+2)%3]) for k in range(3)]; ax = U[:, int(np.argmax(g))]
h = (cv - cv.mean(0)) @ ax
# rim end = open end: fewer points near the extreme (open) -> use the end whose 5%-slab has larger radius spread... use: base end has points near axis (floor)
rad = np.linalg.norm((cv - cv.mean(0)) - np.outer(h, ax), axis=1)
lo, hi = h < np.percentile(h, 8), h > np.percentile(h, 92)
if rad[hi].min() < rad[lo].min(): ax, h = -ax, -h          # floor (points near axis) is at the low end
rim_h = h.max(); centre = cv.mean(0); rim_r = np.median(rad[h > rim_h - 0.01])
sv = mesh['wooden_spoon']; s = sv - sv.mean(0); ws, Us = np.linalg.eigh(s.T @ s); L = Us[:, -1]; t = s @ L
thick = lambda m: np.ptp((s[m]) @ Us[:, 1]) if m.sum() > 5 else 0
bowl = sv[t > np.percentile(t, 85)] if thick(t > np.percentile(t, 85)) > thick(t < np.percentile(t, 15)) else sv[t < np.percentile(t, 15)]
tip = bowl.mean(0)
def measure(Tc, Ts):
    p = Ts[:3, :3] @ tip + Ts[:3, 3]; q = np.linalg.inv(Tc) @ np.r_[p, 1]; v = q[:3] - centre; hh = v @ ax; rr = np.linalg.norm(v - hh * ax)
    return rim_h - hh, rr
def report(label, Tcs, Tss):
    n = len(Tss); ts = np.arange(0, n / 20, 1.0)
    d = [measure(Tcs[int(t * 20)], Tss[int(t * 20)]) for t in ts]
    inside = [(dd > 0 and rr < rim_r) for dd, rr in d]
    print(f'{label:10s} depth cm ' + ' '.join(f'{dd*100:5.1f}' for dd, rr in d)); print(f'{"":10s} radial cm' + ' '.join(f'{rr*100:6.1f}' for dd, rr in d)); print(f'{"":10s} inside  ' + ' '.join('    Y' if x else '    .' for x in inside))
print(f'ep{ep}: cup rim radius {rim_r*100:.1f} cm; t = ' + ' '.join(f'{t:5.0f}' for t in np.arange(0, len(T) / 20, 1.0)))
report('reference', T[:, names.index('blue_cup')], T[:, names.index('wooden_spoon')])
for arg in sys.argv[2:]:
    label, path = arg.split(':', 1)
    if path.endswith('.npz'):
        a = np.load(path); on = list(a['object_names']); Q = a['object_quaternions_wxyz']; Pp = a['object_positions']
        def mk(j):
            M = np.tile(np.eye(4), (len(Pp), 1, 1)); M[:, :3, :3] = R.from_quat(Q[:, j][:, [1, 2, 3, 0]]).as_matrix(); M[:, :3, 3] = Pp[:, j]; return M
        report(label, mk(on.index('blue_cup')), mk(on.index('wooden_spoon')))
    else:
        d = pd.read_parquet(path)
        def mk(nm):
            x = d[d.object_body_name == nm].sort_values('step'); M = np.tile(np.eye(4), (len(x), 1, 1)); M[:, :3, :3] = R.from_quat(x[['achieved_quaternion_x', 'achieved_quaternion_y', 'achieved_quaternion_z', 'achieved_quaternion_w']].to_numpy()).as_matrix(); M[:, :3, 3] = x[['achieved_position_x', 'achieved_position_y', 'achieved_position_z']].to_numpy(); return M
        report(label, mk('blue_cup'), mk('wooden_spoon'))
