#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=P-stageC
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33/logs/stageC-%j.log
# usage: sbatch stage_compose.sbatch ACTOR_DIR BASE_DIR TEACHER_DIR
set -euo pipefail; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; P=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent
cd /tmp; $R/.venv/bin/python -B -u $P/code/stage_ep33_compose_component.py --episode 33 --distilled $1 --base $2 --teacher $3 \
  --teacher-note "open-loop servo teacher: power-grasp pickup (physics_pickup_search_v2, compose world) + teacher ILC (teacher_ilc33.py: cup-axis rim clearance, spoon roll/position ILC); teacher only, not shipped" \
  --lineage-file $P/code/prepare_pickup33.py --lineage-file $P/code/teacher_ilc33.py --lineage-file $P/code/build_c_world.py --lineage-file $3/result.json
