#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=Q25-distill
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25/logs/distill-%j.log
# plain LearnedPolicy grid (unchanged streamF actor_grid_f.py) evaluated against the Stream-Q corrected episode dir.
# usage: sbatch distill_q.sbatch WORLD TEACHER_DIR OUTPUT_DIR [extra actor_grid_f args]
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
W=$1; T=$2; O=$3; shift 3
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamF_ep25/code/actor_grid_f.py --episode 25 --teacher $T --output $O --episode-dir $W/episodes/episode_000025 --workers 12 "$@"
