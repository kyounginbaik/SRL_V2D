#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=e30v9-release
#SBATCH --array=0-11
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v9/logs/release-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=("11.6 0.4 0" "11.6 0.8 0" "11.6 0.4 0.005" "11.6 0.8 0.005" "11.8 0.4 0" "11.8 0.8 0" "11.8 0.4 0.005" "11.8 0.8 0.005" "12.0 0.4 0" "12.0 0.8 0" "12.0 0.4 0.005" "12.0 0.8 0.005")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v9/code/release_v9.py /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v9/retarget/teacher_refit /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v9/episodes/episode_000030 /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/ep30_v9/search/rel_$1_o$2_l$3 $1 $2 $3 0.06 0.06
