#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=P-refit
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33/logs/refit-%A_%a.log
# usage: sbatch --array=0-N refit.sbatch LISTFILE   (each line: BASE OUTNAME args...)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1); set -- $line; BASE=$1; OUT=$2; shift 2
cd /tmp; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $P/code/wrist_refit.py $BASE $P/refit/$OUT "$@"
