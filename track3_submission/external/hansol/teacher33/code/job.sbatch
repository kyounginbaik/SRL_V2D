#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --job-name=T33-job
#SBATCH --output=/data1/hansol/teacher33/logs/%x-%j.log
# usage: E31_TGT=<target dir> sbatch [-c N] [-J ..] job.sbatch SCRIPT.py args...  (SCRIPT in streamK33_ep33/code; node100 because node01 /data1 hung)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/hansol/teacher33/tmp
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
: ${E31_TGT:?E31_TGT unset}; echo "E31_TGT=$E31_TGT H13_ADV=${H13_ADV:-}"
S=$1; shift
cd $TMPDIR; /home/hansol/v2d/.venvs/v2d-host/bin/python -B -u /data1/hansol/teacher33/code/$S "$@" 2>&1 | grep --line-buffered -v -E "Warn|warn\("
