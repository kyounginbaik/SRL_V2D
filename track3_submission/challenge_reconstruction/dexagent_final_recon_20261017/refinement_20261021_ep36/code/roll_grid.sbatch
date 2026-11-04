#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --job-name=ep36-roll
#SBATCH --array=0-5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/logs/roll-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=("1 12.35 12.60 14.65 14.90" "-1 12.35 12.60 14.65 14.90" "1 12.20 12.80 14.55 15.05" "-1 12.20 12.80 14.55 15.05" "1 12.0 12.9 14.5 15.2" "-1 12.0 12.9 14.5 15.2")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/code/roll_replan.py /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/episode_000036/physical_release_search_v2 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/search/roll_$1_$2_$4 $1 $2 $3 $4 $5
