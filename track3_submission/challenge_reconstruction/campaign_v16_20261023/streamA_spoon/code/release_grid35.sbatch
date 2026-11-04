#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:30:00
#SBATCH --job-name=A35-rel
#SBATCH --array=0-5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/logs/rel35-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
G=("18.9 0.3 19.5" "19.0 0.3 19.6" "19.1 0.4 19.8" "18.8 0.5 19.6" "19.2 0.3 19.8" "18.6 0.4 19.4")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $R/.venv/bin/python -B -u $A/code/release_variant.py --teacher $R/refinement_20261021_ep35/search/ret_12.5_18.4_s0.5 --episode-dir $R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter --out $A/ep35/release/rel$1_d$2_lift$3 --t-rel $1 --open-dur $2 --t-lift $3
