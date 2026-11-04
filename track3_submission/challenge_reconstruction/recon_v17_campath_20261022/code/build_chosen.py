"""Apply the public-GT-chosen camera path (avg = mean relative translation + half-way relative rotation of the current SLAM-derived
world_from_rgb and the stereo-PnP path, anchored at frame 0) to the legacy eval episodes. Camera-relative object poses kept.
ep19: the sweeper column is also relabelled by G (half turn, recon_v15 change.json) so the track is consistent with the v14 rows."""
import json, numpy as np, os
from scipy.spatial.transform import Rotation as R, Slerp
O = '/data1/SRL_V2D/challenge_reconstruction/recon_v17_campath_20261022'; F = '/data1/SRL_V2D/challenge_reconstruction/final_recon'
CUR = {36: '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/source_corrected/episode_000036/object_poses.npz'}
G19 = np.asarray(json.load(open('/data1/SRL_V2D/challenge_reconstruction/recon_v15_20261022/episode_000019/change.json'))['G'], float)
rep = {}
for e in (19, 1, 32, 33, 35, 3, 5, 6, 8, 10, 36, 37, 38):
    src = CUR.get(e, f'{F}/episode_{e:06d}/object_poses.npz'); z = np.load(src); names = [str(x) for x in z['names']]
    Ws = z['world_from_rgb']; Two = z['world_from_object']; Wp = np.load(f'{O}/egomotion/eval_ep{e}.npz')['world_from_rgb']
    n = min(len(Ws), len(Wp), len(Two)); anchor = float(np.abs(Ws[0] - Wp[0]).max())
    rs = np.einsum('ij,njk->nik', np.linalg.inv(Ws[0]), Ws[:n]); rp = np.einsum('ij,njk->nik', np.linalg.inv(Wp[0]), Wp[:n])
    half = np.stack([Slerp([0, 1], R.from_matrix(np.stack([rs[k, :3, :3], rp[k, :3, :3]])))(0.5).as_matrix() for k in range(n)])
    rel = np.tile(np.eye(4), (n, 1, 1)); rel[:, :3, :3] = half; rel[:, :3, 3] = 0.5 * (rs[:, :3, 3] + rp[:, :3, 3]); Wa = np.einsum('ij,njk->nik', Ws[0], rel)
    Tco = np.einsum('nij,nmjk->nmik', np.linalg.inv(Ws[:n]), Two[:n]); Tn = np.einsum('nij,nmjk->nmik', Wa, Tco)
    if e == 19: i = names.index('mini_sweeper'); Tn[:, i] = np.einsum('nij,jk->nik', Tn[:, i], G19)
    zz = {k: z[k] for k in z.files}; zz['world_from_object'] = np.concatenate([Tn, Two[n:]], 0); zz['world_from_rgb'] = np.concatenate([Wa, Ws[n:]], 0)
    out = f'{O}/chosen/episode_{e:06d}'; os.makedirs(out, exist_ok=True); np.savez_compressed(f'{out}/object_poses.npz', **zz)
    ch = {nm: float(np.linalg.norm(Tn[:, i, :3, 3] - Two[:n, i, :3, 3], axis=1).mean() * 100) for i, nm in enumerate(names)}
    rep[e] = dict(source=src, frame0_anchor_maxabs=anchor, object_change_mean_cm=ch, relabelled_sweeper=(e == 19)); print(e, 'anchor diff', round(anchor, 6), {k: round(v, 2) for k, v in ch.items()})
json.dump(rep, open(f'{O}/chosen/build_report.json', 'w'), indent=1)
