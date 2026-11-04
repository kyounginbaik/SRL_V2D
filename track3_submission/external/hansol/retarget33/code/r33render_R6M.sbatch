#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /data1/hansol/retarget33/logs/render-%j.log
set -e
R=/data1/hansol/retarget33; O=$R/videos/R6M; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python; export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
RT=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/render_rollouts_titled.py
$PY -B - <<'PYX'
import json, numpy as np
from pathlib import Path
R = Path('/data1/hansol/retarget33'); K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33'); O = R / 'videos/R6M'
B = Path('/data1/hansol/teacher33/runs/save_M1_C_f0/actor/evaluation'); bx = json.load(open(B / 'result.json'))['execution']
A = Path('/data1/hansol/teacher33/runs/save_R6M_C_f0/actor/evaluation'); ex = json.load(open(A / 'result.json'))['execution']
z = np.load(ex['trajectory'], allow_pickle=True); print('keys', z.files[:10])
q = z['qpos'] if 'qpos' in z.files else z['Q']; np.savez(O / 'R6M_trajectory.npz', qpos=q, time_s=np.arange(len(q)) / 20.0)
z2 = np.load(bx['trajectory'], allow_pickle=True); q2 = z2['qpos'] if 'qpos' in z2.files else z2['Q']; np.savez(O / 'M1C_trajectory.npz', qpos=q2, time_s=np.arange(len(q2)) / 20.0)
args = dict(episode_dir='/data1/hansol/teacher33/worlds/R33r6M/ep33/episode_corrected/episode_000033',
            baseline_report=dict(trajectory=str(O / 'M1C_trajectory.npz'), scene=bx['scene']), refined_report=dict(trajectory=str(O / 'R6M_trajectory.npz'), scene=ex['scene']),
            out_dir=str(O), width=480, height=360, stride=1, reference_scene=ex['scene'],
            titles=['Target: r6 recon (moved start)', 'M1_C on c18 (uploaded, 0.90)', 'R6M: r6 moved start, new teacher + full beam'], video_name='ep33_R6M_vs_M1C.mp4')
(O / 'render_args.json').write_text(json.dumps(args, indent=1)); print('ARGS_OK')
PYX
$PY -B $RT --args $O/render_args.json > $O/render.log 2>&1 & P=$!
for i in $(seq 1 150); do [ -f $O/render_report.json ] && break; kill -0 $P 2>/dev/null || break; sleep 5; done; sleep 3; kill $P 2>/dev/null || true; ls -la $O; tail -3 $O/render.log
