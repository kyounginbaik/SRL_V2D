#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35-wrapm
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/logs/wrapm-%j.log
# usage: sbatch wrapm.sbatch DISTILLED_DIR OUT_DIR   (re-apply the ep35 donor margin keys only; evaluates on node01)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
cd /tmp && $R/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/code/wrap_keys.py 35 $1 $2 --keys margin
