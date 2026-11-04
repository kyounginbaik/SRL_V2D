#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=E31-asm
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/logs/assemble-%j.log
# usage: sbatch assemble.sbatch VAR (a|b|acw|bcw) -> finish_v31.py VAR assemble (node100: node01 /data1 mount hung)
set -euo pipefail
cd /tmp
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/code/finish_v31.py $1 assemble
