#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=e30v9-video
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v9/logs/render-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/render_selected.py --episode 30 --baseline /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/search/pickup_s45_lift0.06/release_neutral/trial0/ret_3.2_12.0_s0.75/actor_grid/actor_c1024_p0.003_v0.0001/evaluation --refined /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v9/search_v8scene/rel2_11.6_o1.2_l0_body0.01/actor_grid/actor_c512_p0.001_v5e-05/evaluation --output /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/videos_v9/episode_000030 --stride 1 --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v8/episodes/episode_000030
