#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=node100
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /data1/hansol/retarget33/logs/render-%j.log
set -e
R=/data1/hansol/retarget33; O=$R/videos/R5; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python; export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
$PY -B - <<'PYX'
import json, numpy as np
from pathlib import Path
R = Path('/data1/hansol/retarget33'); K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33'); O = R / 'videos/R5'
for n in ('R2', 'R5'):
    q = np.load(R / f'runs/eval_{n}/eval.npz')['Q']; np.savez(O / f'{n}_trajectory.npz', qpos=q, time_s=np.arange(len(q)) / 20.0)
sc = json.load(open(K / 'runs/save_K33_cwHGAfix_ep33_HGA_c9/actor/evaluation/result.json'))['execution']['scene']
args = dict(episode_dir=str(R / 'worlds/R33_HGA_r6/ep33/episode_corrected/episode_000033'),
            baseline_report=dict(trajectory=str(O / 'R2_trajectory.npz'), scene=sc), refined_report=dict(trajectory=str(O / 'R5_trajectory.npz'), scene=sc),
            out_dir=str(O), width=480, height=360, stride=1, reference_scene=sc,
            titles=['Target: latest recon, start roll -16 (11/03 22:27)', 'R2: uploaded 11/03, MP-SR fail (old recon)', 'R5: retarget on latest recon (package 2b3c46b2)'],
            video_name='ep33_R5_vs_R2.mp4')
(O / 'render_args.json').write_text(json.dumps(args, indent=1)); print('ARGS_OK')
PYX
timeout 1500 $PY -B /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/render_rollouts_titled.py --args $O/render_args.json | tail -3
