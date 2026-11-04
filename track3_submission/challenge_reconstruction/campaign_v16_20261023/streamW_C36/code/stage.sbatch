#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=W36-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamW_C36/logs/%x-%j.log
# usage: sbatch stage.sbatch ACTOR_DIR(evaluated on episode_compose) BASE_DIR(staging_C*) [extra lineage files...]
set -euo pipefail; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; W=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamW_C36
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent
A=$1; BASE=$2; shift 2; EXTRA=""; for f in "$@"; do EXTRA="$EXTRA --lineage-file $f"; done
cd /tmp; $R/.venv/bin/python -B -u $W/code/stage_ep36_compose_component.py --episode 36 --distilled $A --base $BASE \
  --teacher /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep36/release3/seat17.0_rel18.5_lift19.1 \
  --lineage-file $W/code/plan36.py --lineage-file $W/code/cem36.py $EXTRA
