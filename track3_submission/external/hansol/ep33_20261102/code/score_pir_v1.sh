#!/bin/bash
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
W=/data1/hansol/ep33_20261102
/home/hansol/v2d/.venvs/v2d-host/bin/python -B $W/code/score_robot_vs_recon.py $W/xfer/pir_v1_objects_world.npz $W/k33x/targets/HGAR/episode_000033 33 $W/out/pir_v1_score_vs_HGAR.json
