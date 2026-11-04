"""Per-frame image evidence (visible IoU with the source mask, median stereo surface distance) for the current final track vs v14."""
import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
sys.path[:0] = ['/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020/code/postprocess']
import polish as PL, trimesh
from scipy.spatial import cKDTree
FF = PL.FF; F = Path('/data1/SRL_V2D/challenge_reconstruction/final_recon'); V = Path('/data1/SRL_V2D/challenge_reconstruction/recon_v14_20261022')
ep = int(sys.argv[1]); n_ = f'episode_{ep:06d}'; C = json.loads((F / n_ / 'config.json').read_text())
z0 = np.load(F / n_ / 'object_poses.npz'); z1 = np.load(V / n_ / 'object_poses.npz'); cam = z0['world_from_rgb']; names = [str(x) for x in z0['names']]
src = Path(C['source']); k = json.loads((src / 'anycalib/undistorted_intrinsics.json').read_text()); K = np.array([[k['fx'] / 4, 0, k['cx'] / 4], [0, k['fy'] / 4, k['cy'] / 4], [0, 0, 1.]])
obs = PL.Obs(C['depth'], [C['object_masks'][n] for n in names], [str(src / 'masks' / t) for t in C['hand_tracks']])
rep = json.loads((V / n_ / 'report.json').read_text()); out = {}
for o, name in enumerate(names):
    if not rep['objects'][name].get('changed'): continue
    m = trimesh.load(F / n_ / 'assets' / f'{name}.obj', force='mesh', process=False); surf = trimesh.sample.sample_surface(m, 6000, seed=42)[0]; tree = cKDTree(surf)
    rows = {'current': [], 'v14': []}
    for f in range(len(cam)):
        dep, om, hm = obs(f); occ = np.logical_or.reduce(hm + [mm for j, mm in enumerate(om) if j != o]); pts = FF.points_from_mask(dep, om[o] & ~occ, K, 600)
        if len(pts) < 40: continue
        for tag, z in (('current', z0), ('v14', z1)):
            T = np.linalg.inv(cam[f]) @ z['world_from_object'][f, o]; mt, _ = FF.metrics(m.vertices @ T[:3, :3].T + T[:3, 3], m.faces, K, om[o], occ, pts, tree, T); rows[tag].append((f, mt['iou'], mt['depth_median_m']))
    a = np.array(rows['current']); b = np.array(rows['v14'])
    out[name] = dict(frames=len(a), iou_median=[float(np.median(a[:, 1])), float(np.median(b[:, 1]))], depth_median_cm=[float(np.median(a[:, 2]) * 100), float(np.median(b[:, 2]) * 100)],
                     v14_better_frames=float(np.mean((b[:, 1] - 15 * b[:, 2]) > (a[:, 1] - 15 * a[:, 2]))))
    print(n_, name, out[name], flush=True)
(V / n_ / 'evidence_compare.json').write_text(json.dumps(out, indent=1))
