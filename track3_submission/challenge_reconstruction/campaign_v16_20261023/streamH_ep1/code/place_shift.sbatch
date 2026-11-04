#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --job-name=H1-shift
#SBATCH --array=0-11
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/logs/shift-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
H=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1
LID=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep1/lid
# base teacher, dx dy dz (m), t0, t1 (-1 = persist)
G=("r3_12.8_16.7 0.0047 0.0086 0 2.5 12.9" "r3_12.8_16.7 0.0093 0.0171 0 2.5 12.9" "r3_12.8_16.7 0.0140 0.0257 0 2.5 12.9" \
   "r3_12.8_16.7 0.0047 0.0086 0 2.5 -1" "r3_12.8_16.7 0.0093 0.0171 0 2.5 -1" "r3_12.8_16.7 0.0140 0.0257 0 2.5 -1" \
   "r3_12.3_16.7 0.0047 0.0086 0 2.5 12.9" "r3_12.3_16.7 0.0093 0.0171 0 2.5 12.9" "r3_12.3_16.7 0.0140 0.0257 0 2.5 12.9" \
   "r3_12.3_16.7 0.0047 0.0086 0 2.5 -1" "r3_12.3_16.7 0.0093 0.0171 0 2.5 -1" "r3_12.3_16.7 0.0140 0.0257 0 2.5 -1")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp
$PY -u $H/code/place_shift.py $LID/$1 $H/search/place_${1}_${2}_${3}_${4}_${5}_${6} $2 $3 $4 $5 $6
