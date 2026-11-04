#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=node100
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH -o /data1/hansol/retarget33/logs/render-%j.log
set -e
R=/data1/hansol/retarget33; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33; O=$R/videos/R1_c1
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python; export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
$PY -B - <<'PYX'
import json, numpy as np
from pathlib import Path
R = Path('/data1/hansol/retarget33'); K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33'); O = R / 'videos/R1_c1'
q = np.load(R / 'runs/eval_R1_c1/eval.npz')['Q']; np.savez(O / 'R1_c1_trajectory.npz', qpos=q, time_s=np.arange(len(q)) / 20.0)
c9 = json.load(open(K / 'runs/save_K33_cwHGAfix_ep33_HGA_c9/actor/evaluation/result.json'))['execution']
a1 = json.load(open('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13a/runs/save_ep33_b1/actor/evaluation/result.json'))['execution']
assert c9['hashes']['scene'] == a1['hashes']['scene']
args = dict(episode_dir=str(R / 'worlds/R33_HGA7b/ep33/episode_corrected/episode_000033'),
            baseline_report=dict(trajectory=c9['trajectory'], scene=c9['scene']),
            refined_report=dict(trajectory=str(O / 'R1_c1_trajectory.npz'), scene=c9['scene']),
            out_dir=str(O), width=480, height=360, stride=1, reference_scene=c9['scene'],
            titles=['Target: our recon 7b (11/03 15:21)', 'HGA_c9: team retarget, 11/02 recon (official fail)', 'R1_c1: our retarget on 7b'],
            video_name='ep33_R1_c1_vs_HGA_c9.mp4')
(O / 'render_args.json').write_text(json.dumps(args, indent=1)); print('ARGS_OK', q.shape)
PYX
$PY -B /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/render_rollouts_titled.py --args $O/render_args.json | tail -5
