#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=render36v13
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261022_margins/logs/render36-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/render_selected.py --episode 36 --baseline /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/stage_input/margins_c2048/evaluation --refined /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261022_margins/ep36_v12warm_r2/evaluation --output /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/videos_v13/episode_000036 --stride 1 --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261022_margins/render_epdir/episode_000036
