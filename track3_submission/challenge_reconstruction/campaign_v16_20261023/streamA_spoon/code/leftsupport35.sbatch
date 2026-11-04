#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --job-name=A35-ls
#SBATCH --array=0-5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/logs/ls35-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
G=("5.0 5.3 12.5" "6.0 6.3 12.5" "6.0 6.3 17.0" "5.0 5.3 9.5" "4.0 4.3 12.5" "6.3 6.5 10.0")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $R/.venv/bin/python -B -u $A/code/seated_grasp.py --teacher $A/ep35/release/rel19.2_d0.3_lift19.8 --episode-dir $R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter --out $A/ep35/leftsupport/seat$1_$2-$3 --side left --t-seat $1 --t-on $2 --t-off $3 --ramp 0.4
