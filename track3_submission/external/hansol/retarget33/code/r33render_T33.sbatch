#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /data1/hansol/retarget33/logs/render-%j.log
set -e
R=/data1/hansol/retarget33; O=$R/videos/T33; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python; export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
RT=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/render_rollouts_titled.py
$PY -B - <<'PYX'
import json, numpy as np
from pathlib import Path
R = Path('/data1/hansol/retarget33'); K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33'); O = R / 'videos/T33'
sc = json.load(open(K / 'runs/save_K33_cwHGAfix_ep33_HGA_c9/actor/evaluation/result.json'))['execution']['scene']
A = Path('/data1/hansol/teacher33/runs/save_M1_A_f0/actor/evaluation'); ex = json.load(open(A / 'result.json'))['execution']
z = np.load(ex['trajectory'], allow_pickle=True); print('keys', z.files[:10])
q = z['qpos'] if 'qpos' in z.files else z['Q']; np.savez(O / 'T33_trajectory.npz', qpos=q, time_s=np.arange(len(q)) / 20.0)
q2 = np.load(R / 'runs/eval_R12/eval.npz')['Q']; np.savez(O / 'R12_trajectory.npz', qpos=q2, time_s=np.arange(len(q2)) / 20.0)
args = dict(episode_dir=str(R / 'worlds/R33_HGAc18/ep33/episode_corrected/episode_000033'),
            baseline_report=dict(trajectory=str(O / 'R12_trajectory.npz'), scene=sc), refined_report=dict(trajectory=str(O / 'T33_trajectory.npz'), scene=ex['scene']),
            out_dir=str(O), width=480, height=360, stride=1, reference_scene=sc,
            titles=['Target: c18 recon', 'R12: team actor (spoon slips out at 13.5 s)', 'T33 M1_A: new teacher, robust actor'], video_name='ep33_T33_vs_R12.mp4')
(O / 'render_args.json').write_text(json.dumps(args, indent=1)); print('ARGS_OK')
PYX
$PY -B $RT --args $O/render_args.json > $O/render.log 2>&1 & P=$!
for i in $(seq 1 150); do [ -f $O/render_report.json ] && break; kill -0 $P 2>/dev/null || break; sleep 5; done; sleep 3; kill $P 2>/dev/null || true; ls -la $O; tail -3 $O/render.log
