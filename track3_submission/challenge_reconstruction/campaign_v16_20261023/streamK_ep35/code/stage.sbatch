#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/logs/stage-%j.log
# usage: sbatch stage.sbatch ACTOR_DIR BASE_DIR TEACHER_DIR [extra --lineage-file ...]   (one stage job per base dir)
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon
D=$1; BASE=$2; TEACHER=$3; shift 3
cd /tmp && $R/.venv/bin/python -B -u $R/refinement_20261021_ep32/code/stage_3233_component.py --episode 35 --distilled $D --base $BASE --teacher $TEACHER \
  --lineage-file $K/code/ilc_axis.py --lineage-file $A/code/release_variant.py --lineage-file $A/ep35/release/rel19.2_d0.3_lift19.8/result.json \
  --lineage-file $R/refinement_20261021_ep35/search/ret_12.5_18.4_s0.5/result.json --lineage-file $K/README.md "$@"
