#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --job-name=V8-res
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamV8_ep32/logs/res-%j.log
# usage: sbatch [--cpus-per-task=N] res.sbatch SCRIPT.py args...   (SCRIPT relative to streamV8_ep32/code or M8 code)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamV8_ep32/tmp
S=$1; shift; [ -f "$S" ] || S=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamV8_ep32/code/$S
cd $TMPDIR; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $S "$@"
