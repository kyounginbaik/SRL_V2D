#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --job-name=H13b-res
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13b/logs/res-%j.log
# usage: sbatch [-c N] res.sbatch SCRIPT.py args...   (SCRIPT relative to streamH13b/code)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13b/tmp
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13b/code/env.sh
S=$1; shift
cd $TMPDIR; $PY -B -u $H13/code/$S "$@" 2>&1 | grep --line-buffered -v -E "Warn|warn\("
