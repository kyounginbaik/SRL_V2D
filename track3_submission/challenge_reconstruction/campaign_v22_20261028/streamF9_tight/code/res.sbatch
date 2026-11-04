#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node100
#SBATCH --cpus-per-task=48
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=F9-res
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF9_tight/logs/res-%j.log
# usage: sbatch [--nodelist=..] [-c N] res.sbatch SCRIPT.py args...   (SCRIPT relative to streamF9_tight/code)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF9_tight/tmp
S=$1; shift
cd $TMPDIR; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF9_tight/code/$S "$@" 2>&1 | grep --line-buffered -v -E "Warn|warn\("
