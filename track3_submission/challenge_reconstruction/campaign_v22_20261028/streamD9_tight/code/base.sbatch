#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=D9-base
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamD9_tight/logs/base-%j.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamD9_tight/code/env.sh
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
cd $D9/tmp
for e in 3 6 19 25; do eval A=\$A$e; eval M=\$M$e
  $PY -B -u $D9/code/mres9.py eval --ep $e --actor $A --meshes $M --out $D9/runs/ep$e/base_eval > $D9/logs/base_ep$e.log 2>&1 &
done; wait
