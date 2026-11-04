#!/bin/bash
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=2
cd /tmp
/home/hansol/v2d/.venvs/v2d-host/bin/python -B /data1/hansol/ep33_20261102/code/compare_ep33.py /data1/hansol/ep33_20261102/out/ep33_compare --fps 10
