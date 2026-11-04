#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=G36-marg
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep36_margins-%j.log
# usage: sbatch ep36_margins.sbatch ACTOR_DIR OUT_DIR SEED   (fresh bounded finger-margin CEM on top of a plain LearnedPolicy, refinement_20261022_margins tool)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
cd $R/code && $R/.venv/bin/python -B -u $R/refinement_20261022_margins/code/search_finger_margins_gen.py --episode 36 --actor $1 --output $2 --samples 96 --iterations 12 --workers 16 --bound 0.25 --seed $3
