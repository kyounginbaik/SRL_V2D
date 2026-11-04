#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --job-name=F19-port
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/logs/F19-port-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# usage: sbatch --array=0-N port.sbatch JOBFILE   (each line: TEACHER OUT ARGS...)
L=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1)
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/code/port19.py $L
