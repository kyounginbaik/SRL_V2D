"""Episode 10: real beige-in-blue insertion from stereo depth (end frames), vs our tracking reference and the shipped physics."""
import numpy as np, cv2, json, trimesh, pandas as pd
from scipy.spatial.transform import Rotation as R
F = '/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000010'; S = '/data1/SRL_V2D/challenge_reconstruction/eval_stereo_current_20261010/episode_000010'
MK = '/data1/SRL_V2D/challenge_submissions/track3/eval_hawor_latest_20261008/outputs/episode_000010'
z = np.load(F + '/object_poses.npz'); Wr = z['world_from_rgb']; To = z['world_from_object']; names = [str(x) for x in z['names']]
K = np.load(F + '/camera_motion.npz')['K']
def axis_height(v):
    c = v - v.mean(0); w, U = np.linalg.eigh(c.T @ c); g = [abs(w[i]-w[(i+1)%3]) + abs(w[i]-w[(i+2)%3]) - abs(w[(i+1)%3]-w[(i+2)%3]) for i in range(3)]
    a = U[:, int(np.argmax(g))]; p = v @ a; return float(p.max() - p.min()), a
ours = {n: np.asarray(trimesh.load(f'{F}/assets/{n}.obj', force='mesh').vertices) for n in names}
scan = {n: np.asarray(trimesh.load(f'/data1/SRL_V2D/gt_best_per_object/{n}.ply', force='mesh').vertices) for n in names}
H = {n: dict(ours=axis_height(ours[n])[0] * 100, scan=axis_height(scan[n])[0] * 100) for n in names}
print('cup heights cm (along symmetry axis):', {n: {k: round(v, 1) for k, v in h.items()} for n, h in H.items()})
d = pd.read_parquet('/data1/SRL_V2D/challenge_submissions/track3_v8_20261021/all_20_episodes/results/episode_000010.parquet'); g = {n: x.sort_values('step') for n, x in d.groupby('object_body_name')}
def mesh_world(n, k, kind):
    if kind == 'ref': Tm = To[k, names.index(n)]; return ours[n] @ Tm[:3, :3].T + Tm[:3, 3]
    x = g[n].iloc[k]; p = x[[f'achieved_position_{a}' for a in 'xyz']].to_numpy(float); q = x[[f'achieved_quaternion_{a}' for a in 'xyzw']].to_numpy(float); return ours[n] @ R.from_quat(q).as_matrix().T + p
H_, W_ = 760, 1014; uu, vv = np.meshgrid(np.arange(W_), np.arange(H_)); rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1)
rows = []
for k in range(130, 161, 5):
    enc = cv2.imread(f'{S}/depth_rgb/{k:06d}.png', cv2.IMREAD_UNCHANGED).astype(float); dep = 65535. / np.maximum(enc, 1) - 1; dep[enc >= 65535] = 0
    pts = {}
    for n in names:
        m = cv2.imread(f'{MK}/{n}/masks/1/{k:06d}.png', 0); m = cv2.resize(m, (W_, H_), interpolation=cv2.INTER_NEAREST) > 0
        m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0; sel = m & (dep > 0.1) & (dep < 1.5)
        P = rays[sel] * dep[sel][:, None]; P = P @ Wr[k, :3, :3].T + Wr[k, :3, 3]; pts[n] = P
    allm = np.zeros((H_, W_), bool)
    for n in names: allm |= cv2.resize(cv2.imread(f'{MK}/{n}/masks/1/{k:06d}.png', 0), (W_, H_), interpolation=cv2.INTER_NEAREST) > 0
    allm = cv2.dilate(allm.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    P = rays[(~allm) & (dep > 0.1)] * dep[(~allm) & (dep > 0.1)][:, None]; P = P @ Wr[k, :3, :3].T + Wr[k, :3, 3]
    c = np.concatenate([pts['beige_cup'], pts['blue_cup']]).mean(0); near = P[np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < 0.15]; table = float(np.median(near[:, 2])) if len(near) else np.nan
    top = float(np.percentile(pts['beige_cup'][:, 2], 98)); r = dict(frame=k, table_z_cm=table * 100, beige_top_depth_cm=top * 100, n_beige=len(pts['beige_cup']), n_blue=len(pts['blue_cup']),
        blue_top_depth_cm=float(np.percentile(pts['blue_cup'][:, 2], 98)) * 100 if len(pts['blue_cup']) else np.nan)
    for kind in ('ref', 'phys'):
        bw = mesh_world('beige_cup', min(k, len(g['beige_cup']) - 1), kind); blw = mesh_world('blue_cup', min(k, len(g['blue_cup']) - 1), kind)
        r[f'{kind}_beige_top_cm'] = float(bw[:, 2].max()) * 100; r[f'{kind}_blue_bottom_cm'] = float(blw[:, 2].min()) * 100; r[f'{kind}_insertion_cm'] = float(blw[:, 2].max() - bw[:, 2].min()) * 100
    rows.append(r)
df = pd.DataFrame(rows); pd.set_option('display.width', 250); print(df.round(2).to_string(index=False))
stack_depth = df.beige_top_depth_cm - df.table_z_cm
for which in ('ours', 'scan'):
    ins = H['blue_cup'][which] + H['beige_cup'][which] - stack_depth
    print(f'real insertion from depth, using {which} cup heights: median {ins.median():.1f} cm (per frame {np.round(ins.values,1).tolist()})')
print('reference insertion median %.1f cm, physics %.1f cm' % (df.ref_insertion_cm.median(), df.phys_insertion_cm.median()))
print('stack height above table: depth %.1f | ref %.1f | phys %.1f cm' % (stack_depth.median(), (df.ref_beige_top_cm - df.table_z_cm).median(), (df.phys_beige_top_cm - df.table_z_cm).median()))
json.dump(dict(rows=rows, heights=H), open('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep10/diag/insertion_measure.json', 'w'), indent=1, default=float)
