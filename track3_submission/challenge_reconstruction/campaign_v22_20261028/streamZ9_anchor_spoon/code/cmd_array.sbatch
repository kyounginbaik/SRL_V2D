#!/bin/bash
#SBATCH --job-name=Z9-anchor
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:50:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/logs/cmd-%A_%a.log
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /data1/SRL_V2D
LIST=${LIST:-challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code/cmds.txt}
STRIDE=${STRIDE:-5}
for i in $(seq 0 $((STRIDE-1))); do
  LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID*STRIDE+i+1))p" "$LIST")
  [ -z "$LINE" ] && continue
  eval "$LINE" > /dev/null || echo "FAIL $LINE"
  echo "DONE $LINE"
done
