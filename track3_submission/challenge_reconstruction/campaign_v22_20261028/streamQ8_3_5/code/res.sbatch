#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --job-name=Q8-res
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamQ8_3_5/logs/res-%j.log
# usage: sbatch [-c N] res.sbatch SCRIPT.py args...  (SCRIPT relative to streamQ8_3_5/code)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamQ8_3_5/tmp
S=$1; shift
cd $TMPDIR; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamQ8_3_5/code/$S "$@" 2>&1 | grep --line-buffered -v -E "Warn|warn\("
