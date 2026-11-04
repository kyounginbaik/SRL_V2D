#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=F19-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/logs/F19-stage-%j.log
# usage: sbatch stage.sbatch DISTILLED_ACTOR_DIR BASE TEACHER WORLD_SUFFIX JOBFILE
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
F=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19
cd /tmp && $R/.venv/bin/python -B -u $F/code/stage_ep19F_component.py --episode 19 --episode-dir $F/world$4/episode_target/episode_000019 \
  --distilled $1 --base $2 --teacher $3 \
  --lineage-file $F/code/port19.py --lineage-file $F/code/build_world19.py --lineage-file $F/code/common19F.py --lineage-file $F/code/twin_rest.py \
  --lineage-file $F/twin_poses.npz --lineage-file $5 --lineage-file $F/world$4/scene/scene_change.json \
  --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py
