"""Cross-check of ep36 targets in the CANONICAL (stereo) camera: marker-ball median px and IoU vs canonical masks (guard_clip Evidence),
per 0.5 s window, for FINAL (target_final_v14_g17), tgt_s1, tgt_s4. Shows whether the RGB-silhouette fits agree with the stereo evidence."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/home/hansol/v2d/outputs/ep32_robot_20261031/code'); sys.path.insert(0, '/home/hansol/v2d/outputs/spoon_stage_audit_20261029/code')
from ep32_robot_vs_video import CleanBalls, DELTA
from ep32_late_roll import ball_px
from guard_clip import Evidence, CAN
from retime import shift
E = 'episode_000036'; C22 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028')
BUN = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17') / E
Told = np.load(C22 / 'target_final_v14_g17' / E / 'wooden_spoon_trajectory.npz')['T_world_obj']
z = np.load(BUN / 'object_poses.npz'); s = [str(n) for n in z['names']].index('wooden_spoon'); camF = z['camera_from_object'][:, s]; F = len(camF)
root = CAN / 'evaluation' / E; dd = DELTA['evaluation_36']; dd = np.asarray(dd, float)[:F] if np.ndim(dd) else np.full(F, float(dd))
I = json.loads((root / 'intrinsics.json').read_text()); K = np.array([[I['fx'], 0, I['cx']], [0, I['fy'], I['cy']], [0, 0, 1.]]); balls = CleanBalls(root, F)
mj = json.loads((BUN / 'meshes.json').read_text()); ev = Evidence(root, ['wooden_spoon'], {'wooden_spoon': BUN / mj['wooden_spoon']['file']}, F)
T = {'final': Told}
for k in ("tgt_s1", "tgt_s4", "tgt_s5"): T[k] = np.load(f'/data1/hansol/spoon3537/targets/{k}/{E}/wooden_spoon_trajectory.npz')['T_world_obj']
fr = list(range(20, F - 1)); res = {}
for k, X in T.items():
    cb = shift(np.einsum('tij,tjk->tik', camF, np.einsum('tij,tjk->tik', np.linalg.inv(Told), X)), -dd)
    iou = ev.eval('wooden_spoon', cb, fr)[0]; bp = np.array([ball_px(balls.model, cb[f], K, balls.det[f]) if balls.ok and f < len(balls.det) else np.nan for f in range(F)])
    res[k] = (iou, bp)
print('window | IoU canonical final/s1/s4/s5 | ball px final/s1/s4/s5')
for a in np.arange(1, 20, 0.5):
    f = [t for t in fr if a * 20 <= t < (a + 0.5) * 20]
    if not f: continue
    io = [np.nanmean(res[k][0][f]) for k in T]; bp = [np.nanmedian(res[k][1][f]) if np.isfinite(res[k][1][f]).any() else np.nan for k in T]
    print('%5.1f | %s | %s' % (a, ' '.join('%.2f' % x for x in io), ' '.join('%5.1f' % x for x in bp)))
for lo, hi in ((40, 200), (250, 350)):
    print('frames %d-%d  IoU %s  ball median %s' % (lo, hi, {k: round(float(np.nanmean(res[k][0][lo:hi])), 3) for k in T}, {k: round(float(np.nanmedian(res[k][1][lo:hi])), 1) for k in T}))
