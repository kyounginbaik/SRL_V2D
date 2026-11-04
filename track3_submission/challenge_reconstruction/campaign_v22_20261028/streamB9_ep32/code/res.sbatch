#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --job-name=B9-res
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB9_ep32/logs/res-%j.log
# usage: sbatch [--cpus-per-task=N --nodelist=X] res.sbatch SCRIPT.py args...   (SCRIPT relative to streamB9_ep32/code)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB9_ep32/tmp
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
S=$1; shift; [ "${S:0:1}" = / ] || S=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB9_ep32/code/$S
cd $TMPDIR; case $S in *.sh) bash $S "$@";; *) /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $S "$@";; esac
