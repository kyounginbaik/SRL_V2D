#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=ep35-ret2
#SBATCH --array=0-2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/logs/ret2-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=("14.5 18.4 0.5 111" "5.0 12.5 0.5 112" "16.0 18.4 0.75 113"); set -- ${G[$SLURM_ARRAY_TASK_ID]}
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/code/retention.py --source /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/search/ret_12.5_18.4_s0.5 --episode-dir /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter --output /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/search/s2_ret_$1_$2_s$3 --start-s $1 --stop-s $2 --protect-object wooden_spoon --seed $4 --scale $3 --samples 96 --iterations 12 --workers 16
