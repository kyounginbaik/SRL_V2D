"""K33-HG copy of K32 rollcmp: per-frame kit-faithful errors + signed spoon roll (about the target mesh's long PCA axis, after the kit registration and frame-0
scene alignment) of SUBMITTED ep32 rows (package parquet, f0 mesh payload) vs $E31_TGT. usage: rollcmp.py OUT.npz LABEL=PKG_DIR ..."""
import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import h13lib as L
from scipy.spatial.transform import Rotation as R
EP = 33; names = L.NAMES[EP]; g = L.geometry(EP); tgt = L.target(EP); out = {}
v = g['wooden_spoon']['v']; c = v - v.mean(0); w, U = np.linalg.eigh(c.T @ c); LONG = U[:, 2] * np.sign(U[np.argmax(np.abs(U[:, 2])), 2])
for arg in sys.argv[2:]:
    lab, pkg = arg.split('=', 1)
    s = pd.read_parquet(Path(pkg) / 'track3_submission.parquet'); s = s[s.row_id.str.startswith(f'track_3/e{EP:06d}/')].copy()
    s['f'] = s.row_id.str.extract(r'/f(\d+)/')[0].astype(int); s['o'] = s.row_id.str.extract(r'/o(\d+)$')[0].astype(int)
    res = pd.read_parquet(Path(pkg) / f'results/episode_{EP:06d}.parquet', columns=['object_id', 'object_body_name']).drop_duplicates().sort_values('object_id')
    slot = {n: int(res.object_id[res.object_body_name == n].iloc[0]) for n in names}
    reg = {}; poses = {}
    for i, n in enumerate(names):
        r = s[s.o == slot[n]].sort_values('f')
        cand = L.M._decode_object_mesh(r.object_mesh_b64.iloc[0])
        reg[n] = L.M._register_object_body(cand, (g[n]['v'], g[n]['f']), g[n]['anchor'], 1_000_003 * EP + 101 * i)
        T = np.tile(np.eye(4), (len(r), 1, 1)); T[:, :3, :3] = R.from_quat(r[['quat_x', 'quat_y', 'quat_z', 'quat_w']].values).as_matrix(); T[:, :3, 3] = r[['pos_x', 'pos_y', 'pos_z']].values
        poses[n] = T
    N = min(len(poses[names[0]]), len(tgt[names[0]]))
    ach = np.stack([L.T2pq(poses[n][:N]) for n in names], 1)[:, None]; ref = np.stack([L.T2pq(tgt[n][:N]) for n in names], 1)[:, None].copy()
    axes = []
    for k, n in enumerate(names):
        ca, rot = reg[n]; ach[:, :, k] = L.M._convert_object_body(ach[:, :, k], ca, rot)
        ref[:, :, k, :3] += np.einsum('...ij,j->...i', L.M._rotation_matrix(ref[:, :, k, 3:]), g[n]['anchor']); axes.append(g[n]['axis'])
    al = L.M._align_registered_scene(ach, ref, axes)
    e = L.per_frame(EP, {n: poses[n] for n in names}, reg, tgt)
    k = names.index('wooden_spoon')
    Ra = R.from_quat(al[:, 0, k, 3:]); Rr = R.from_quat(ref[:, 0, k, 3:]); d = (Rr.inv() * Ra).as_rotvec()   # body-frame residual (target mesh frame)
    roll = np.degrees(d @ LONG); other = np.degrees(np.linalg.norm(d - np.outer(d @ LONG, LONG), axis=1))
    out[f'{lab}_cup_cm'], out[f'{lab}_cup_deg'] = e['blue_cup']; out[f'{lab}_sp_cm'], out[f'{lab}_sp_deg'] = e['wooden_spoon']
    out[f'{lab}_sp_roll'] = roll; out[f'{lab}_sp_tilt'] = other
    segs = [(a, a + 40) for a in range(0, 320, 40)] + [(320, 348)]
    print(lab, 'mean cup %.2f/%.1f spoon %.2f/%.1f' % (e['blue_cup'][0].mean(), e['blue_cup'][1].mean(), e['wooden_spoon'][0].mean(), e['wooden_spoon'][1].mean()))
    print('   spoon roll  ', ' '.join(f'f{a}-{b}:{np.median(roll[a:b]):+.0f}' for a, b in segs))
    print('   spoon tilt  ', ' '.join(f'f{a}-{b}:{np.median(other[a:b]):.0f}' for a, b in segs))
    print('   spoon deg   ', ' '.join(f'f{a}-{b}:{e["wooden_spoon"][1][a:b].mean():.0f}' for a, b in segs))
    print('   spoon cm    ', ' '.join(f'f{a}-{b}:{e["wooden_spoon"][0][a:b].mean():.1f}' for a, b in segs), flush=True)
np.savez(sys.argv[1], **out)
