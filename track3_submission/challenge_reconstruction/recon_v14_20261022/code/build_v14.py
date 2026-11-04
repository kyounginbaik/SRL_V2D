"""Build recon_v14 object tracks for one eval episode: objects with a metric-mesh FoundationPose rerun get
  metric FP (object->camera, same undistorted ego_cam_a frame) -> world via the final camera poses
  -> twin continuity (initial twin = closest to the current final track at frame 0)
  -> metric stereo/silhouette polish (refine.py residual + acceptance) -> light smoothing.
Other objects keep the current final track. Writes recon_v14_20261022/episode_X/object_poses.npz + report.json.
usage: build_v14.py EPISODE [--no-polish]"""
import sys, os, json, glob, numpy as np, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020/code/postprocess']
import pp, polish as PL
V = Path('/data1/SRL_V2D/challenge_reconstruction/recon_v14_20261022'); F = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon')
ep = int(sys.argv[1]); nopol = '--no-polish' in sys.argv; n_ = f'episode_{ep:06d}'
C = json.loads((F / n_ / 'config.json').read_text()); z = np.load(F / n_ / 'object_poses.npz'); cam = z['world_from_rgb']; Tcur = z['world_from_object']; names = [str(x) for x in z['names']]
src = Path(C['source']); k = json.loads((src / 'anycalib/undistorted_intrinsics.json').read_text()); K = np.array([[k['fx'] / 4, 0, k['cx'] / 4], [0, k['fy'] / 4, k['cy'] / 4], [0, 0, 1.]])
obs = PL.Obs(C['depth'], [C['object_masks'][n] for n in names], [str(src / 'masks' / t) for t in C['hand_tracks']])
out = Tcur.copy(); rep = dict(episode=n_, objects={})
for o, name in enumerate(names):
    fs = sorted(glob.glob(str(V / 'fp_metric' / n_ / f'poses_{name}' / '*.json')))
    if len(fs) != len(cam): rep['objects'][name] = dict(changed=False, reason='no complete metric FP rerun'); continue
    Tco = np.tile(np.eye(4), (len(fs), 1, 1))
    for i, f in enumerate(fs):
        j = json.load(open(f)); q = np.asarray(j['rotation'], float); Tco[i, :3, :3] = R.from_quat(q[[1, 2, 3, 0]]).as_matrix(); Tco[i, :3, 3] = j['translation']
    Tw = cam @ Tco; mesh = str(F / n_ / 'assets' / f'{name}.obj'); tw, c, w = pp.principal_twins(mesh)
    init = int(np.argmin([pp.rot_dist(Tcur[0, o, :3, :3], (Tw[0] @ S)[:3, :3]) for S in tw]))
    Tf, picks = pp.continuity(Tw, tw, c, init=init)
    Tp, acc = (Tf, np.zeros(len(Tf), bool)) if nopol else PL.polish(Tf, cam, K, mesh, obs, o)
    Ts = pp.smooth(Tp, c, half=2); out[:, o] = Ts
    cen = lambda T: np.einsum('nij,j->ni', T[:, :3, :3], c) + T[:, :3, 3]
    dp = np.linalg.norm(cen(Ts) - cen(Tcur[:, o]), axis=1) * 100; dr = np.array([pp.rot_dist(a[:3, :3], b[:3, :3]) for a, b in zip(Ts, Tcur[:, o])])
    rep['objects'][name] = dict(changed=True, init_twin=init, twin_frames=int((picks != init).sum()), polish_accept=float(acc.mean()),
        delta_vs_current=dict(pos_mean_cm=float(dp.mean()), pos_max_cm=float(dp.max()), rot_mean_deg=float(dr.mean()), rot_max_deg=float(dr.max())))
    print(n_, name, rep['objects'][name], flush=True)
(V / n_).mkdir(parents=True, exist_ok=True)
np.savez_compressed(V / n_ / 'object_poses.npz', world_from_object=out, world_from_rgb=cam, frames=z['frames'], names=z['names'])
(V / n_ / 'report.json').write_text(json.dumps(rep, indent=1))
