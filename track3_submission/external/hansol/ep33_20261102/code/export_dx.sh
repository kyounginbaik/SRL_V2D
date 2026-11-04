#!/bin/bash
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
P="/home/hansol/v2d/.venvs/v2d-host/bin/python -B /data1/hansol/ep33_20261102/code/ep33_export_for_dexagent.py"
$P /data1/hansol/ep33_20261102/xfer/ep33/bundle/episode_000033 /data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033 33 /data1/hansol/ep33_20261102/xfer/ep33_export_HGA.npz
$P /data1/hansol/ep33_20261102/xfer/ep33/bundle/episode_000033_HGB /data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000033 33 /data1/hansol/ep33_20261102/xfer/ep33_export_HGB.npz
