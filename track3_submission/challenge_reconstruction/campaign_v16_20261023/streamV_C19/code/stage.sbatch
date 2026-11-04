#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=v19-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamV_C19/logs/stage-%j.log
# usage: sbatch stage.sbatch DISTILLED_ACTOR_DIR BASE TEACHER
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
V=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamV_C19
cd /tmp && $R/.venv/bin/python -B -u $V/code/stage_ep19C_component.py --episode 19 --episode-dir $V/staging_C/episode_compose_swk/episode_000019 \
  --distilled $1 --base $2 --teacher $3 \
  --lineage-file $V/code/port_relative.py --lineage-file $V/code/build_corrected.py --lineage-file $V/code/common19.py \
  --lineage-file $V/code/port_jobs13.txt --lineage-file $V/staging_C/scene_swk/scene_change.json \
  --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py
