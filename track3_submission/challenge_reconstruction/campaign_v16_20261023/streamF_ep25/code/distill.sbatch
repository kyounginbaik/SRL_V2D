#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=F25-distill
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamF_ep25/logs/distill-%j.log
# usage: sbatch distill.sbatch TEACHER_DIR OUTPUT_DIR [extra actor_grid_f args]
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
T=$1; O=$2; shift 2
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamF_ep25/code/actor_grid_f.py --episode 25 --teacher $T --output $O --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/physics_slots_v3/episodes/episode_000025 --workers 12 "$@"
