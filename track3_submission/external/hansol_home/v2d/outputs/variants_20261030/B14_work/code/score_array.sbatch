#!/bin/bash
#SBATCH --job-name=B14-score
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hansol/v2d/outputs/variants_20261030/B14_work/logs/score-%A_%a.log
# Variant B: public-only scoring of export folders in our mirror (GT enters only the unchanged bscore via score_bundle.py)
set -uo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python
cd /data1/SRL_V2D
LIST=${LIST:?}
STRIDE=${STRIDE:-2}
METRICS=${METRICS:-}
for i in $(seq 0 $((STRIDE-1))); do
  LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID*STRIDE+i+1))p" "$LIST")
  [ -z "$LINE" ] && continue
  $PY -B tools/reconstruction_audit/score_bundle.py "$LINE" $METRICS > /dev/null || echo "FAIL $LINE"
  echo "DONE $LINE"
done
