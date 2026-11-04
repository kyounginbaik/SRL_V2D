#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --job-name=rack30-lift
#SBATCH --array=0-11
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/logs/lift30-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
G=("0.02 2.5 3.5" "0.04 2.5 3.5" "0.06 2.5 3.5" "0.08 2.5 3.5" "0.02 2.3 3.5" "0.04 2.3 3.5" "0.06 2.3 3.5" "0.04 2.2 3.5" "0.04 2.5 4.0" "0.06 2.5 4.0" "0.08 2.5 4.0" "0.06 2.3 4.0")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/code/extra_lift.py /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/retarget/episode_000030 /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021/search/episode_000030/lift_$1_$2_$3 $1 $2 $3
