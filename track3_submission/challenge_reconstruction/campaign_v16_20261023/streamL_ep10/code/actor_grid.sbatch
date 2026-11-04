#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=L10-actors
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamL_ep10/logs/actors-%j.log
# usage: sbatch --export=ALL,SRC=<teacher dir with result.json> actor_grid.sbatch  -> SRC/actor_grid
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SRC:?}"
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; S=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamL_ep10
cd /tmp && $R/.venv/bin/python -B -u $S/code/actor_grid_L.py --episode 10 --source $SRC --output $SRC/actor_grid --workers 8
