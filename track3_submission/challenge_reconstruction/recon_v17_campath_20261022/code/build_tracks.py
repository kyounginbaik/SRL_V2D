"""Per legacy episode: compare the current world tracks (SLAM-path world_from_rgb) with the same camera-relative object
poses placed through the stereo-PnP camera path; GT-free static-object cue decides; write recon_v17 tracks."""
import json, numpy as np, os, sys
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, os.path.dirname(__file__)); from static_cue import rest_mask, jitter
O = '/data1/SRL_V2D/challenge_reconstruction/recon_v17_campath_20261022'; F = '/data1/SRL_V2D/challenge_reconstruction/final_recon'
CUR = {36: '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/source_corrected/episode_000036/object_poses.npz'}
rows = {}
for e in (1, 3, 5, 6, 8, 10, 19, 32, 33, 35, 36, 37, 38):
    src = CUR.get(e, f'{F}/episode_{e:06d}/object_poses.npz'); z = np.load(src); names = [str(x) for x in z['names']]
    Two = z['world_from_object']; Ws = z['world_from_rgb']; P = np.load(f'{O}/egomotion/eval_ep{e}.npz'); Wp = P['world_from_rgb']; inl = P['inliers']
    n = min(len(Ws), len(Wp), len(Two)); Ws, Wp, Two = Ws[:n], Wp[:n], Two[:n]
    rs = np.einsum('ij,njk->nik', np.linalg.inv(Ws[0]), Ws); rp = np.einsum('ij,njk->nik', np.linalg.inv(Wp[0]), Wp)
    cd = np.linalg.norm(rs[:, :3, 3] - rp[:, :3, 3], axis=1) * 100; ca = np.degrees((R.from_matrix(rs[:, :3, :3]) * R.from_matrix(rp[:, :3, :3]).inv()).magnitude())
    Tco = np.einsum('nij,mnjk->mnik', np.linalg.inv(Ws), np.moveaxis(Two, 1, 0))          # (obj, n, 4, 4)
    Tnew = np.einsum('nij,mnjk->mnik', Wp, Tco)
    objs = {}; votes = []
    for i, nm in enumerate(names):
        Ts = np.moveaxis(Two, 1, 0)[i]; Tp = Tnew[i]; d = np.linalg.norm(Ts[:, :3, 3] - Tp[:, :3, 3], axis=1) * 100
        m = rest_mask(Ts, Tp); js, nr = jitter(Ts, m); jp, _ = jitter(Tp, m)
        objs[nm] = dict(change_mean_cm=float(d.mean()), change_max_cm=float(d.max()), rest_frames=int(nr), static_jitter_slam_cm=js, static_jitter_pnp_cm=jp)
        if nr >= 20 and np.isfinite(js) and np.isfinite(jp): votes.append(jp < js * 0.9)          # PnP must be >=10% stiller
    healthy = bool((inl[1:] < 40).sum() <= 3 and np.percentile(inl[1:], 5) >= 80)
    material = any(o['change_mean_cm'] > 1.0 for o in objs.values())
    cue = 'pnp' if votes and all(votes) else ('slam' if votes else 'undetermined')
    use = healthy and material and cue == 'pnp'
    rows[e] = dict(source=src, camera_travel_cm=dict(slam=float(np.linalg.norm(rs[:, :3, 3], axis=1).max() * 100), pnp=float(np.linalg.norm(rp[:, :3, 3], axis=1).max() * 100)),
                   camera_diff=dict(mean_cm=float(cd.mean()), max_cm=float(cd.max()), mean_deg=float(ca.mean()), max_deg=float(ca.max())),
                   pnp_inliers=dict(min=int(inl[1:].min()), p5=int(np.percentile(inl[1:], 5)), median=int(np.median(inl[1:])), weak_frames=int((inl[1:] < 40).sum())),
                   objects=objs, healthy=healthy, material=material, static_cue=cue, use_pnp=use)
    out = f'{O}/episode_{e:06d}'; os.makedirs(out, exist_ok=True)
    zz = {k: z[k] for k in z.files}
    if use:
        zz['world_from_object'] = np.concatenate([np.moveaxis(Tnew, 0, 1), Two.copy()[n:] if len(z['world_from_object']) > n else np.zeros((0,) + Two.shape[1:])], 0) if len(z['world_from_object']) > n else np.moveaxis(Tnew, 0, 1)
        zz['world_from_rgb'] = Wp if len(z['world_from_rgb']) == n else np.concatenate([Wp, z['world_from_rgb'][n:]], 0)
    np.savez_compressed(f'{out}/object_poses.npz', **zz)
    r = rows[e]; print(f"ep{e:2d} travel slam {r['camera_travel_cm']['slam']:5.1f} pnp {r['camera_travel_cm']['pnp']:5.1f} | cam diff mean/max {r['camera_diff']['mean_cm']:.2f}/{r['camera_diff']['max_cm']:.2f} cm {r['camera_diff']['max_deg']:.2f} deg | inl min/p5/med {r['pnp_inliers']['min']}/{r['pnp_inliers']['p5']}/{r['pnp_inliers']['median']} weak {r['pnp_inliers']['weak_frames']} | " +
          ' '.join(f"{k}: {v['change_mean_cm']:.2f}/{v['change_max_cm']:.2f}cm rest{v['rest_frames']} jit {v['static_jitter_slam_cm']:.2f}->{v['static_jitter_pnp_cm']:.2f}" for k, v in r['objects'].items()) + f" | cue {cue} -> {'PnP' if use else 'keep'}")
json.dump(rows, open(f'{O}/camera_path_eval.json', 'w'), indent=1, default=float)
