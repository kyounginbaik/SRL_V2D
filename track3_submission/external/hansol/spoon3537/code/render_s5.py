"""Package A videos: original | FINAL target (kinematic) | FINAL robot | package A robot, for ep36 (s1d) and ep37 (s1a)."""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code')
from render_rollouts_titled import render_comparison
C22 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamALL20K/staging')
R = Path('/data1/hansol/spoon3537/rt/runs'); O = Path('/data1/hansol/spoon3537/work/videosA'); O.mkdir(parents=True, exist_ok=True)
for ep, comp, new, lab in ((36, 'tgt_v32_ep36_b2', 'save_tgt_s5_ep36_s5', 'ep36 s5 (15 s depth jump fixed, 14.5-16.5 s re-searched)'),):
    E = f'episode_{ep:06d}'; C = C22 / comp / 'components' / E
    epd = Path(json.load(open(R / new / 'actor/evaluation/result.json'))['input_dir'])
    q = np.load(C / 'portable/original/evaluation_trajectory.npz', allow_pickle=False)
    qq = q['qpos'] if 'qpos' in q.files else q['Q']; bt = O / f'ep{ep}_FINAL_trajectory.npz'; np.savez(bt, qpos=qq, time_s=np.arange(len(qq)) / 20.0)
    ex = json.load(open(R / new / 'actor/evaluation/result.json'))['execution']
    z = np.load(ex['trajectory'], allow_pickle=True); qn = z['qpos'] if 'qpos' in z.files else z['Q']; nt = O / f'ep{ep}_s5_trajectory.npz'; np.savez(nt, qpos=qn, time_s=np.arange(len(qn)) / 20.0)
    print(ep, 'frames', len(qq), len(qn))
    render_comparison(str(epd), dict(trajectory=str(bt), scene=str(C / 'portable/original/scene_physics.xml')), dict(trajectory=str(nt), scene=ex['scene']),
                      str(O), width=480, height=360, stride=1, reference_scene=str(C / 'portable/original/scene_physics.xml'),
                      titles=('Target: FINAL recon (kinematic)', 'FINAL robot', lab), video_name=f'ep{ep}_s5_vs_FINAL.mp4')
    print('VIDEO', O / f'ep{ep}_s5_vs_FINAL.mp4')
