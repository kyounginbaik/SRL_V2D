#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=E31-job
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE31_cowork2/logs/%x-%j.log
# usage: E31_TGT=<target dir> sbatch [-c N] [-J ..] job.sbatch SCRIPT.py args...  (SCRIPT in streamE31_cowork2/code; node100 because node01 /data1 hung)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE31_cowork2/tmp
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
: ${E31_TGT:?E31_TGT unset}; echo "E31_TGT=$E31_TGT H13_ADV=${H13_ADV:-}"
S=$1; shift
cd $TMPDIR; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE31_cowork2/code/$S "$@" 2>&1 | grep --line-buffered -v -E "Warn|warn\("
