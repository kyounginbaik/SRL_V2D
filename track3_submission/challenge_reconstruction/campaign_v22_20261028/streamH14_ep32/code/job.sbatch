#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=10
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --job-name=H14-job
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH14_ep32/logs/%x-%j.log
# usage: sbatch [--cpus-per-task N] [--job-name ..] job.sbatch SCRIPT.py args...  (SCRIPT in streamH14_ep32/code; env H13_ADV passed through)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH14_ep32/tmp
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
S=$1; shift
cd $TMPDIR; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH14_ep32/code/$S "$@" 2>&1 | grep --line-buffered -v -E "Warn|warn\("
