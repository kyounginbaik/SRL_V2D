#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --job-name=C9-res
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC9_tight/logs/res-%j.log
# usage: sbatch [--nodelist=..] [--cpus-per-task=N] res.sbatch SCRIPT.py args...   (SCRIPT in streamC9_tight/code)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC9_tight/tmp
S=$1; shift
cd $TMPDIR; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC9_tight/code/$S "$@"
