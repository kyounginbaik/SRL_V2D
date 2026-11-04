#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --job-name=A35-seat
#SBATCH --array=0-9
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/logs/seat35-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
G=("13.5 0.7" "15.0 0.7" "16.5 0.7" "18.0 0.7" "15.0 1.3" "16.5 1.3" "15.0 2.5" "16.5 4.0" "15.0 0.7 --rot-only" "16.5 1.3 --rot-only")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $R/.venv/bin/python -B -u $A/code/seated_grasp.py --teacher $A/ep35/release/rel19.2_d0.3_lift19.8 --episode-dir $R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter --out $A/ep35/seat/seat$1_on$2$3 --t-seat $1 --t-on $2 $3
