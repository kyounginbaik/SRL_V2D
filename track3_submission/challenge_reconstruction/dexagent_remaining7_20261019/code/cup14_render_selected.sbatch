#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=cup14-render
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/cup14-render-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
py=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
actor="$base/diagnostics/cup_transfer/release_neutral/trial2/actor_grid/actor_c1024_p0.001_v5e-05/evaluation"
"$py" -u "$base/code/render_selected.py" --episode 14 --baseline "$base/physics_collision_v2/runs/baseline_v1/episode_000014/baseline" --refined "$actor" --output "$actor/videos" --stride 1
