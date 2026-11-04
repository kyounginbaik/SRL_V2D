#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=ANC-audit
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamANCHOR/logs/%x-%A_%a.log
# usage: sbatch --array=1,3,... audit.sbatch
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp && python3 -I /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamANCHOR/code/anchor_audit.py $SLURM_ARRAY_TASK_ID
