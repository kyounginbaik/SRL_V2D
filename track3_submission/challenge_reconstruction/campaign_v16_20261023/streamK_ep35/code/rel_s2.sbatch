#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35-rel
#SBATCH --array=0-7
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/logs/rel-%A_%a.log
# release re-plans (Stream A release_variant.py) on the pass-2 retention teachers of refinement_20261021_ep35
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/code; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35
EPD=$R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter; S=$R/refinement_20261021_ep35/search
G=("s2_ret_16.0_18.4_s0.75 19.2 0.3 19.8" "s2_ret_16.0_18.4_s0.75 19.0 0.3 19.6" "s2_ret_16.0_18.4_s0.75 18.9 0.3 19.5" "s2_ret_16.0_18.4_s0.75 19.1 0.4 19.8" "s2_ret_14.5_18.4_s0.5 19.2 0.3 19.8" "s2_ret_14.5_18.4_s0.5 19.0 0.3 19.6" "s2_ret_5.0_12.5_s0.5 19.2 0.3 19.8" "s2_ret_5.0_12.5_s0.5 19.0 0.3 19.6")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -B -u $A/release_variant.py --teacher $S/$1 --episode-dir $EPD --out $K/teachers/release/$1_rel$2_d$3_lift$4 --t-rel $2 --open-dur $3 --t-lift $4 --blend-dur 0.6
