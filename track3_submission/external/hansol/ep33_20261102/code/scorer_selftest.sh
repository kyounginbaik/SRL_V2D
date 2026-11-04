#!/bin/bash
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102
$PY - <<'PYX'
import numpy as np
x=np.load('/data1/hansol/ep33_20261102/xfer/ep33_export_HGA.npz')
np.savez('/data1/hansol/ep33_20261102/xfer/selftest_robot.npz', names=x['names'], world_from_object=x['world_from_object_rgb'], clock='rgb', fps=20.0)
PYX
$PY $W/code/score_robot_vs_recon.py $W/xfer/selftest_robot.npz $W/k33x/targets/HGAR/episode_000033 33 $W/xfer/selftest_score.json
