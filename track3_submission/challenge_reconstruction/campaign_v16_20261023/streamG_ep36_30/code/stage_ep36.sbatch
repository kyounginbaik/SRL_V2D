#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=G36-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/stage36-%j.log
# usage: sbatch stage_ep36.sbatch DISTILLED_ACTOR_DIR BASE_DIR TEACHER_DIR [extra --lineage-file ...]
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
D=$1; BASE=$2; TEACHER=$3; shift 3
cd /tmp && $R/.venv/bin/python -B -u $R/refinement_20261021_ep36/code/stage_ep36_component.py --episode 36 --distilled $D --base $BASE --teacher $TEACHER --lineage-file $R/refinement_20261021_ep36/code/roll_replan.py --lineage-file $R/refinement_20261021_ep36/source_corrected/episode_000036/spoon_track_correction_v9.json "$@"
