#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --job-name=spoon-retention
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/spoon-retention-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
legacy=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
name=$(printf 'episode_%06d' "$SPOON_EPISODE")
"$legacy/.venv/bin/python" -u "$legacy/code/refine_expanded_retention.py" --source "$base/diagnostics/spoon_transfer/$name/$SPOON_SOURCE" --episode-dir "$base/physics_collision_v2/episodes/$name" --output "$base/diagnostics/spoon_transfer/$name/$SPOON_OUTPUT" --start-s "$SPOON_START" --stop-s "$SPOON_STOP" --protect-object plastic_dish_rack --seed "$SPOON_SEED" --scale "$SPOON_SCALE" --samples 96 --iterations 12 --workers 16
