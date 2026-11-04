#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=K8-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK8_35_36_37/logs/%x-%j.log
# usage: sbatch stage.sbatch EP ACTOR_DIR BASE_DIR TEACHER_DIR [extra lineage files...]
set -euo pipefail; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; W=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK8_35_36_37
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent
EP=$1; A=$2; BASE=$3; T=$4; shift 4; EXTRA=""; for f in "$@"; do EXTRA="$EXTRA --lineage-file $f"; done
cd /tmp; $R/.venv/bin/python -B -u $W/code/stageK.py --episode $EP --distilled $A --base $BASE --teacher $T \
  --lineage-file $W/code/planK.py --lineage-file $W/code/cemK.py --lineage-file $W/code/ilcK.py --lineage-file $W/code/tlibK.py $EXTRA
