#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=L8-res
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamL8_3_5_6/logs/res-%j.log
# usage: sbatch [--nodelist=..] res.sbatch SCRIPT.py args...   (AMD nodes only: node01 / node100)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamL8_3_5_6/tmp
S=$1; shift
cd $TMPDIR; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamL8_3_5_6/code/$S "$@"
