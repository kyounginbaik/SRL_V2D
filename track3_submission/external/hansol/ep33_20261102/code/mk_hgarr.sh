#!/bin/bash
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"
$PY /data1/hansol/ep33_20261102/code/ep33_restroll_target.py /data1/hansol/ep33_20261102/roll/recon_E14_best_B2A_S1_HGAR8s15/episode_000033 /data1/hansol/ep33_20261102/k33x/targets/HGAR_rest2/episode_000033 rest 25 40 --bundle-out /data1/hansol/ep33_20261102/roll/recon_E14_best_B2A_S1_HGARR/episode_000033
$PY /data1/hansol/ep33_20261102/code/ep33_export_for_dexagent.py /data1/hansol/ep33_20261102/roll/recon_E14_best_B2A_S1_HGARR/episode_000033 /data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033 33 /data1/hansol/ep33_20261102/xfer/ep33_export_HGARR.npz
