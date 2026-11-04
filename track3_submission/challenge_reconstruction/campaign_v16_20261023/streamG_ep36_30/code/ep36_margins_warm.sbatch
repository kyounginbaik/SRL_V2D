#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=G36-margw
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep36_marginsw-%j.log
# usage: sbatch ep36_margins_warm.sbatch PLAIN_ACTOR_DIR NAME SEED ITER   (wrap with the v13 donor margins as warm start, then fresh bounded margin CEM)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; G=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep36
W=$G/wrapped_probe/$2_s1.0
cd /tmp; [ -f $W/evaluation/result.json ] || $R/.venv/bin/python -B -u $R/refinement_20261021_ep36/code/wrap_margins.py $1/training/policy.npz $W 1.0
cd $R/code && $R/.venv/bin/python -B -u $R/refinement_20261022_margins/code/search_finger_margins_gen.py --episode 36 --actor $W --output $G/margins/$2_warmdonor --samples 96 --iterations $4 --workers 16 --bound 0.25 --seed $3
