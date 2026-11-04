"""Score a common_v2 run's achieved spoon poses (setup frame) against our references in the same setup frame (anchor point,
full orientation; achieved mapped into the source frame by the frame-0 anchor, exact because achieved frame 0 == source frame 0): v21 marker-corrected reference, original final_recon, compose-retargeted.
usage: vs_refs.py RUN_EP_DIR"""
import sys, json, numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as R
run = sys.argv[1]; r = pd.read_parquet(run + '/results/episode_000036.parquet'); r = r[r.environment_id == 0].sort_values('reference_frame')
E = '/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/episodes/episode_000036'
S = np.array(json.load(open(E + '/input_manifest.json'))['setup_from_source'])
anch = np.array(json.load(open(E + '/metric_geometry.json'))['objects']['wooden_spoon']['anchor_local_m'])
pa = r[[f'achieved_position_{a}' for a in 'xyz']].values; Ra = R.from_quat(r[[f'achieved_quaternion_{a}' for a in 'xyzw']].values).as_matrix()
pr0 = r[[f'reference_position_{a}' for a in 'xyz']].values
refs = dict(v21_corrected='/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/source_corrected/episode_000036/object_poses.npz',
            original='/data1/SRL_V2D/challenge_reconstruction/final_recon/episode_000036/object_poses.npz',
            compose='/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamW_C36/source_compose/episode_000036/object_poses.npz')
# the results parquet frame: check it equals the setup frame (reference columns == S @ compose source)
Ta = np.tile(np.eye(4), (len(pa), 1, 1)); Ta[:, :3, :3] = Ra; Ta[:, :3, 3] = pa
O0 = np.load(refs['original'])['world_from_object'][0, 0]; Ta = (O0 @ np.linalg.inv(Ta[0])) @ Ta; pa, Ra = Ta[:, :3, 3], Ta[:, :3, :3]   # frame-0 anchor (achieved f0 == source f0 by construction)
for k, p in refs.items():
    T = np.load(p)['world_from_object'][:, 0]
    pe = np.linalg.norm(pa + np.einsum('tij,j->ti', Ra, anch) - (T[:, :3, 3] + np.einsum('tij,j->ti', T[:, :3, :3], anch)), axis=1) * 100
    re = np.degrees(R.from_matrix(np.einsum('tji,tjk->tik', T[:, :3, :3], Ra)).magnitude())
    print(f'{k:14s} {pe.mean():.3f} cm {re.mean():.2f} deg  ')
