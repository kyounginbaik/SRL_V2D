#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --job-name=A35-seatb
#SBATCH --array=0-5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/logs/seatb35-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
G=("13.0 13.2 18.8" "14.0 14.2 18.8" "6.3 6.5 12.5" "6.3 6.5 18.8" "9.0 9.2 18.8" "16.0 16.2 18.8")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $R/.venv/bin/python -B -u $A/code/seated_grasp.py --teacher $A/ep35/release/rel19.2_d0.3_lift19.8 --episode-dir $R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter --out $A/ep35/seat/right_seat$1_$2-$3 --side right --t-seat $1 --t-on $2 --t-off $3 --ramp 0.6
