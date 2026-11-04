#!/bin/bash
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
cd /tmp && OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} $PY -B "$@" 2>&1 | grep -v Warn
