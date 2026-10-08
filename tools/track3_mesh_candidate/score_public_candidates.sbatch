#!/bin/bash
#SBATCH --job-name=t3-score-candidates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/logs/score_candidates_%j.out
#SBATCH --error=/data1/SRL_V2D/challenge_reconstruction/public_mesh_benchmark/logs/score_candidates_%j.err
set -euo pipefail
code="${V2D_MESH_CODE_DIR:-/data1/SRL_V2D/tools/track3_mesh_candidate}"
/home/exaFLOPs/miniforge3/bin/python3 "$code/score_public_candidates.py"
