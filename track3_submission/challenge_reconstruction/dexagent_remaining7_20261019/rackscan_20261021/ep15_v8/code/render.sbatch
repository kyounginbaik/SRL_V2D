#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=v8video15
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/logs/video-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/render_selected.py --episode 15 --baseline /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/spoon_transfer/episode_000015/retention_window32_44/actor_grid/actor_c1024_p0.003_v0.0001/evaluation --refined /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/search/episode_000015/s3_ret_3.2_4.8_s0.25 --output /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/videos_v8/episode_000015_best_teacher_not_actor --stride 1 --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep15_v8/episodes/episode_000015
