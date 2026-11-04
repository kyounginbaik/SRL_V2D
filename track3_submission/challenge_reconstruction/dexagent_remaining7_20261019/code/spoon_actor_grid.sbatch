#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=spoon-actor-grid
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/spoon-actor-grid-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
legacy=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
name=$(printf 'episode_%06d' "$SPOON_EPISODE")
"$legacy/.venv/bin/python" -u "$base/code/spoon_actor_grid.py" --episode "$SPOON_EPISODE" --source "$base/diagnostics/spoon_transfer/$name/$SPOON_SOURCE" --output "$base/diagnostics/spoon_transfer/$name/$SPOON_SOURCE/actor_grid" --workers 4
