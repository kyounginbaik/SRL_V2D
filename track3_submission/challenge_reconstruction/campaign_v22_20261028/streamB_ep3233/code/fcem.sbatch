#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=B3233-fcem
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB_ep3233/logs/fcem-%A_%a.log
# usage: sbatch --array=0-N fcem.sbatch LIST  (line: fcem.py args)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1)
cd /tmp; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB_ep3233/code/${FC:-fcem.py} $line
