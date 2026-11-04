import json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, '/home/hansol/v2d/outputs/ep32_robot_20261031/code')
sys.path.insert(0, '/home/hansol/v2d/outputs/spoon_stage_audit_20261029/code')
from ep32_robot_vs_video import DELTA
C22 = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028')
comps = {35: 'streamK35_ep35/staging/K35_H2joint_b1/components/episode_000035', 36: 'streamALL20K/staging/tgt_v32_ep36_b2/components/episode_000036', 37: 'streamALL20K/staging/tgt_v32_ep37_b1/components/episode_000037'}
tg = {35: C22 / 'streamK35_ep35/targets/K35_H2joint/episode_000035', 36: C22 / 'target_final_v14_g17/episode_000036', 37: C22 / 'target_final_v14_g17/episode_000037'}
for ep, c in comps.items():
    E = f'episode_{ep:06d}'; PV = C22 / c / 'portable_validation'
    man = json.loads((PV / f'runtime/episodes/{E}/input_manifest.json').read_text()); src = Path(man['source'])
    op = np.load(src / 'object_poses.npz'); print(ep, 'source', src, 'keys', op.files, 'video', man.get('source_video'), 'fps', man['fps'], 'frames', man['frames'])
    print('  setup_from_source', np.round(np.array(man['setup_from_source']), 3).tolist())
    T = np.load(tg[ep] / 'wooden_spoon_trajectory.npz'); print('  target keys', T.files, {k: T[k].shape for k in T.files})
    Tw = T['T_world_obj']; W = op['world_from_object'][:, list(op['names']).index('wooden_spoon')]
    n = min(len(Tw), len(W)); print('  target vs source wfo max diff', float(np.abs(Tw[:n] - W[:n]).max()), 'len', len(Tw), len(W))
    d = DELTA.get(f'evaluation_{ep}'); print('  DELTA', None if d is None else (np.round(np.asarray(d)[:5], 3).tolist() if np.ndim(d) else d))
    print('  ls src', sorted(p.name for p in src.iterdir())[:30])
