#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:50:00
#SBATCH --job-name=A9-reg
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamA9_meshswap/logs/reg-%A_%a.log
set -uo pipefail
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
cd /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamA9_meshswap/code
EPS=(1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38)
EP=${EPS[$SLURM_ARRAY_TASK_ID]}
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B regcheck.py ../out/audit_ep$EP.json ../out/reg_ep$EP.json
