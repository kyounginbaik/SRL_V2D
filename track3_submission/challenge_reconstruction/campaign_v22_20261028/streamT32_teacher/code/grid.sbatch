#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:45:00
#SBATCH --job-name=B3233-grid
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB_ep3233/logs/grid-%A_%a.log
# usage: sbatch --array=0-N grid.sbatch LIST   (line: SCRIPT ARGS...; script relative to streamB code/ or absolute)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1); set -- $line; S=$1; shift
[[ $S = /* ]] || S=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB_ep3233/code/$S
cd /tmp; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $S "$@"
