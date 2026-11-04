#!/bin/bash
#SBATCH -p debug --nodelist=node100 -c 4 --mem=32G -t 00:50:00 -J C30-asm -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/logs/assemble_%j.log
# usage: sbatch assemble.sbatch K
set -euo pipefail; cd /tmp
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python; X=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose
$PY -B -u $X/code/finish_v30c.py $1 assemble
$PY -B $X/code/mkzip.py /data1/SRL_V2D/challenge_submissions_SRL/track3_v30c$1_20261101
