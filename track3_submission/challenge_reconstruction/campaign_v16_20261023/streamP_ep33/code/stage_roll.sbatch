#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=P-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33/logs/stage-%j.log
# usage: sbatch stage_roll.sbatch ACTOR_DIR(evaluated on the roll-corrected episode dir) BASE_DIR
set -euo pipefail; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; P=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent
cd /tmp; $R/.venv/bin/python -B -u $P/code/stage_ep33_roll_component.py --episode 33 --distilled $1 --base $2 \
  --teacher $R/refinement_20261021_ep33/search/v12_segE/c2_R_9.8_12.3 --lineage-file $P/code/wrist_refit.py --lineage-file $P/code/roll_refit.py ${3:+--lineage-file $3} ${4:+--lineage-file $4}
