"""ep36 tgt_s6 = FINAL target + s5 jump fix (f297-308) + roll-only refit 12.5-17.5 s (spoon_annot_roll3, RGB f250-350, free end, ball x3,
IoU term, face-thr 0.25) with the agent marks, EXCEPT f298-312 set to edge-on [0, 1] (concave side down toward the table) = USER'S VISUAL
JUDGEMENT (a human hint, to be disclosed). Acceptance (canonical/stereo camera): ball median f250-350 <= FINAL's, canonical IoU drop <= 0.02.
Writes tgt_s6 atomically only if accepted. stage: prep | final"""
import sys, json, shutil, os, hashlib, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
sys.path.insert(0, '/home/hansol/v2d/outputs/ep32_robot_20261031/code'); sys.path.insert(0, '/home/hansol/v2d/outputs/spoon_stage_audit_20261029/code')
from ep32_robot_vs_video import CleanBalls, DELTA
from ep32_late_roll import ball_px
from guard_clip import Evidence, CAN
from retime import shift
E = 'episode_000036'; C22 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028')
BUN = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17') / E
TGT = C22 / 'target_final_v14_g17' / E; S5 = Path('/data1/hansol/spoon3537/targets/tgt_s5') / E
W = Path('/data1/hansol/spoon3537/work/s6'); W.mkdir(parents=True, exist_ok=True)
Told = np.load(TGT / 'wooden_spoon_trajectory.npz')['T_world_obj'].astype(np.float64); T5 = np.load(S5 / 'wooden_spoon_trajectory.npz')['T_world_obj'].astype(np.float64)
z = dict(np.load(BUN / 'object_poses.npz')); s = [str(n) for n in z['names']].index('wooden_spoon'); F = len(z['camera_from_object'])
stage = sys.argv[1]
if stage == 'prep':
    mk = json.loads(Path('/data1/hansol/spoon3537/am36/agent_marks36.json').read_text()); n_agent = len(mk)
    for f in range(298, 313): mk[str(f)] = [0.0, 1.0]
    Path(W / 'marks36_s6.json').write_text(json.dumps(mk, indent=0))
    (W / 'marks36_s6_PROVENANCE.json').write_text(json.dumps(dict(agent_marks='/data1/hansol/spoon3537/am36/agent_marks36.json', agent_marks_count=n_agent,
        human_hint=dict(frames='RGB f298-312 (14.9-15.6 s)', value='edge-on [0, 1] = concave side pointing down in the image, toward the table',
                        source="user's visual judgement relayed by the coordinator (human hint, to be disclosed)"))))
    Rel5 = np.einsum('tij,tjk->tik', np.linalg.inv(Told), T5); bd = W / 'bundle' / E
    if bd.exists(): shutil.rmtree(bd)
    shutil.copytree(BUN, bd)
    for key in ('world_from_object', 'camera_from_object'):
        a = z[key].copy(); a[:, s] = np.einsum('tij,tjk->tik', a[:, s], Rel5); z[key] = a
    np.savez_compressed(bd / 'object_poses.npz', **z); print('PREP OK', bd)
    sys.exit(0)
# final
zb = np.load(W / 'bundle' / E / 'object_poses.npz'); zr = np.load(W / 'roll/bundle' / E / 'object_poses.npz')
Rr = np.einsum('tij,tjk->tik', np.linalg.inv(zb['world_from_object'][:, s]), zr['world_from_object'][:, s])
T6 = np.einsum('tij,tjk->tik', T5, Rr); assert np.allclose(T6[0], Told[0])
root = CAN / 'evaluation' / E; dd = DELTA['evaluation_36']; dd = np.asarray(dd, float)[:F] if np.ndim(dd) else np.full(F, float(dd))
I = json.loads((root / 'intrinsics.json').read_text()); K = np.array([[I['fx'], 0, I['cx']], [0, I['fy'], I['cy']], [0, 0, 1.]]); balls = CleanBalls(root, F)
mj = json.loads((BUN / 'meshes.json').read_text()); ev = Evidence(root, ['wooden_spoon'], {'wooden_spoon': BUN / mj['wooden_spoon']['file']}, F)
camF = np.load(BUN / 'object_poses.npz')['camera_from_object'][:, s]; fr = list(range(250, 351)); out = {}
for k, X in (('FINAL', Told), ('s5', T5), ('s6', T6)):
    cb = shift(np.einsum('tij,tjk->tik', camF, np.einsum('tij,tjk->tik', np.linalg.inv(Told), X)), -dd)
    io = ev.eval('wooden_spoon', cb, fr)[0]; bp = np.array([ball_px(balls.model, cb[f], K, balls.det[f]) if balls.ok and f < len(balls.det) else np.nan for f in fr])
    win = {f'{a:.1f}': [round(float(np.nanmean(io[[f for f in fr if a * 20 <= f < (a + 0.5) * 20]])), 3),
                        round(float(np.nanmedian(bp[[i for i, f in enumerate(fr) if a * 20 <= f < (a + 0.5) * 20]])), 1) if np.isfinite(bp[[i for i, f in enumerate(fr) if a * 20 <= f < (a + 0.5) * 20]]).any() else None]
           for a in np.arange(12.5, 17.5, 0.5)}
    out[k] = dict(iou=float(np.nanmean(io[fr])), ball_median=float(np.nanmedian(bp)), per_0p5s_iou_ball=win)
ang = np.degrees(R.from_matrix(Rr[:, :3, :3]).magnitude()); roll_per_s = {f'{t / 20:.1f}': round(float(ang[t]), 1) for t in range(240, F, 10)}
ok = out['s6']['ball_median'] <= out['FINAL']['ball_median'] + 1e-9 and out['s6']['iou'] >= out['FINAL']['iou'] - 0.02
rep = dict(accepted=bool(ok), eval_f250_350=out, roll_change_deg_by_time=roll_per_s, max_roll_change_deg=float(ang.max()),
           human_hint='RGB f298-312 edge-on [0,1] (concave down toward the table) = user visual judgement; disclose', roll_fit_summary=open(W / 'roll.txt').read().strip().splitlines()[-1][:800])
if ok:
    OUT = Path('/data1/hansol/spoon3537/targets/tgt_s6') / E; TMP = Path('/data1/hansol/spoon3537/targets/.tmp_s6') / E
    if TMP.parent.exists(): shutil.rmtree(TMP.parent)
    shutil.copytree(TGT, TMP, symlinks=False); np.savez_compressed(TMP / 'wooden_spoon_trajectory.npz', T_world_obj=T6)
    (TMP / 'HUMAN_HINT_NOTE.json').write_text(json.dumps(dict(note='roll fit 12.5-17.5 s used agent marks plus a human visual hint at RGB f298-312 (edge-on, concave down)',
                                                              provenance=str(W / 'marks36_s6_PROVENANCE.json'))))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists(): shutil.rmtree(OUT)
    os.rename(TMP, OUT); shutil.rmtree(TMP.parent); rep['target'] = str(OUT); rep['sha'] = hashlib.sha256((OUT / 'wooden_spoon_trajectory.npz').read_bytes()).hexdigest()
else:
    np.savez(W / 'T6_rejected.npz', T6=T6)
(W / 's6_report.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep))
