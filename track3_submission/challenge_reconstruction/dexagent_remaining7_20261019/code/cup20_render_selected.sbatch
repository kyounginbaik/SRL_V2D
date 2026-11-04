#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --job-name=cup20-render
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/cup20-render-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2
py=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
"$py" -u "/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/render_selected.py" --episode 20 --baseline "/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_collision_v2/runs/baseline_v1/episode_000020/baseline" --refined "/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/episode20_hold/final_pitcher_release_75_neutral/trial1/actor_grid/actor_c1024_p0.003_v0.0001/evaluation" --output "/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/episode20_hold/final_pitcher_release_75_neutral/trial1/actor_grid/actor_c1024_p0.003_v0.0001/evaluation/videos" --stride 1
