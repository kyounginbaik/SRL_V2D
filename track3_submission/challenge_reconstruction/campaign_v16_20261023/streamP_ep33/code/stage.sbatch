#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=P-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33/logs/stage-%j.log
# usage: sbatch stage.sbatch ACTOR_DIR BASE_DIR   (one job per base dir; original reference, unchanged scene/meshes)
set -euo pipefail; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; P=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33
J=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233
export PYTHONPATH=$R/refinement_20261021_ep32/code:$R/code:/data1/SRL_V2D/third_party/DexAgent
cd /tmp; $R/.venv/bin/python -B -u $R/refinement_20261021_ep32/code/stage_3233_component.py --episode 33 --distilled $1 --base $2 \
  --teacher $R/refinement_20261021_ep33/search/v12_segE/c2_R_9.8_12.3 \
  --lineage-file $J/ep33/staging/components/episode_000033/staged.json --lineage-file $P/code/wrist_refit.py --lineage-file $P/audit/original.json
