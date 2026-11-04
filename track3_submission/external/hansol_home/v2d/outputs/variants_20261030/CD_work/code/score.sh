#!/bin/bash
# Run the team's UNCHANGED public scorer (tools/reconstruction_audit/score_bundle.py; GT only inside bscore) on one export folder
# with the teammate venv site-packages (python 3.11, pandas 2.3.3, numpy 1.26.4) under our own python 3.11.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
cd /data1/SRL_V2D
/home/hansol/v2d/.venvs/v2d-host/bin/python -B tools/reconstruction_audit/score_bundle.py "$1" --metrics > /dev/null || echo "FAIL $1"
