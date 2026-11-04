#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=P8-chain
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamP8_ep35/logs/%x-%j.log
# usage: sbatch --nodelist=node200 chain.sbatch BASE_ACTOR NAME 'TERMS_JSON' WINDOWS  -> runs/bake_NAME/actor, runs/i_NAME
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamP8_ep35/tmp
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; S=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamP8_ep35
echo host=$(hostname) args="$@"
cd $R/code && $R/.venv/bin/python -B -u $S/code/bake.py $1 $S/runs/bake_$2/actor "$3" 2>&1 | grep -E "search|FINAL" && \
timeout 2600 $R/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK8_35_36_37/code/ilcK.py ilc --ep 35 --actor $S/runs/bake_$2/actor --out $S/runs/i_$2 --windows $4 --iterations 5 --workers 32 --aps 0,0.3,0.6,1.0,1.5 --ars 0,0.3,0.6,1.0,1.5
