#!/bin/bash
# env wrapper: our python 3.11 + team venv site-packages (same libs as the team's qc.py / scorer runs)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
exec /home/hansol/v2d/.venvs/v2d-host/bin/python -B "$@"
