#!/bin/bash
#SBATCH --job-name=Z9-score
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/logs/score-%A_%a.log
# R8: public-only scoring (GT enters only bscore) of variant folders; several lines per task (STRIDE)
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /data1/SRL_V2D
LIST=${LIST:-challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code/score_list.txt}
STRIDE=${STRIDE:-4}
for i in $(seq 0 $((STRIDE-1))); do
  LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID*STRIDE+i+1))p" "$LIST")
  [ -z "$LINE" ] && continue
  /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B tools/reconstruction_audit/score_bundle.py "$LINE" --metrics > /dev/null || echo "FAIL $LINE"
  echo "DONE $LINE"
done
