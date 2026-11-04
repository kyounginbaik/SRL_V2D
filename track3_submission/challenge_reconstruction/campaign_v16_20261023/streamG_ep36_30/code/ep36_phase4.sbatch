#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --job-name=G36-p4
#SBATCH --array=0-2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep36_p4-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep36; A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/code
EPD=$R/refinement_20261021_ep36/episode_corrected/episode_000036; cd /tmp
rel() { $PY -B -u $A/release_variant.py --teacher $1 --episode-dir $EPD --out $2 --t-rel $3 --open-dur 0.3 --t-lift $4 --blend-dur 0.6; }
case $SLURM_ARRAY_TASK_ID in
 0) rel $G/ilc2/seat17.5_gr0.5_gp0.3/iter0 $G/release3/seat17.5ilc_rel18.3_lift18.9 18.3 18.9;;
 1) rel $G/ilc2/seat17.5_gr0.5_gp0.3/iter0 $G/release3/seat17.5ilc_rel18.4_lift19.0 18.4 19.0;;
 2) rel $G/ilc3/seat17.5_gr0.3_gp0.3/iter1 $G/release3/seat17.5ilc03b_rel18.5_lift19.1 18.5 19.1;;
esac
