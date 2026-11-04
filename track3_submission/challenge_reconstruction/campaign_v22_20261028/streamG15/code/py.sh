#!/bin/bash
cd /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG15/code && OMP_NUM_THREADS=2 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B "$@"
