#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35-margins
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/logs/margins-%j.log
# usage: sbatch margins.sbatch ACTOR_DIR OUT_DIR SEED [ITER=12] [BOUND=0.25] [TOL_DEG=2.0]
# resumable fresh finger-margin CEM (right hand + left pinky) over a plain/wrapped actor; re-submit with the same args to continue
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35
cd $R/code && timeout 2820 $R/.venv/bin/python -B -u $K/code/search_margins_k.py --episode 35 --actor $1 --output $2 --samples 96 --iterations ${4:-12} --workers 16 --bound ${5:-0.25} --seed $3 --tolerance-deg ${6:-2.0} --left-pinky --resume
