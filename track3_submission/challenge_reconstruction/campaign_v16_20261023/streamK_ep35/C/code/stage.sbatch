#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35C-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/C/logs/stage-%j.log
# usage: sbatch stage.sbatch ACTOR_DIR BASE_DIR TEACHER_DIR [extra --lineage-file ...]   (one stage job per base dir)
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35
D=$1; BASE=$2; TEACHER=$3; shift 3
cd /tmp && $R/.venv/bin/python -B -u $K/C/code/stage_c35_component.py --episode 35 --distilled $D --base $BASE --teacher $TEACHER \
  --lineage-file $K/C/code/build_target.py --lineage-file $K/C/code/build_episode.py --lineage-file $K/code/ilc_axis.py \
  --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/code/retention.py \
  --lineage-file $K/C/target/compose_v1.json "$@"
