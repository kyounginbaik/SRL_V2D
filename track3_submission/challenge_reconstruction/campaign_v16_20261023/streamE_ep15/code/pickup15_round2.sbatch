#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --job-name=E-pick15r2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamE_ep15/logs/pick15r2-%A_%a.log
# usage: sbatch --array=0-N pickup15_round2.sbatch   with pickup_round2/jobs.txt lines: NAME SPEC RESUME_PROGRESS SEED
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamE_ep15
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $W/pickup_round2/jobs.txt); set -- $line
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $W/code/rack_safe_pickup.py --spec $2 --output $W/pickup_round2/$1 --resume $3 --seed $4 --samples 96 --iterations 12 --workers 8
