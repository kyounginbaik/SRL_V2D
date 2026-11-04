#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=K8-marg
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK8_35_36_37/logs/%x-%j.log
# usage: sbatch [--nodelist] margins.sbatch ACTOR_DIR OUT_DIR SEED [ITER=12] [BOUND=0.25] [TOL_DEG=2.0]   (streamK_ep35/code/search_margins_k.py unchanged, resumable;
# the actor's evaluation input_dir = C7 target_final_v5 world, so its cost is vs target_final_v5)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35
echo host=$(hostname) "$@"
cd $R/code && timeout 2820 $R/.venv/bin/python -B -u $K/code/search_margins_k.py --episode 35 --actor $1 --output $2 --samples 96 --iterations ${4:-12} --workers 16 --bound ${5:-0.25} --seed $3 --tolerance-deg ${6:-2.0} --left-pinky --resume
