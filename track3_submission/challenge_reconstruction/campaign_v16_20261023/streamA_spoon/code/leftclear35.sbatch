#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --job-name=A35-lc
#SBATCH --array=0-4
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/logs/lc35-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
G=("0.06 0 3.0 9.5" "0.10 0 3.0 9.5" "0.08 0 2.5 12.0" "0.08 0.06 3.0 9.5" "0.12 0 3.0 12.0")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $R/.venv/bin/python -B -u $A/code/wrist_offset.py --teacher $A/ep35/release/rel19.2_d0.3_lift19.8 --episode-dir $R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter --out $A/ep35/leftclear/dz$1_dy$2_t$3-$4 --side left --dz $1 --dy $2 --t-on $3 --t-off $4
