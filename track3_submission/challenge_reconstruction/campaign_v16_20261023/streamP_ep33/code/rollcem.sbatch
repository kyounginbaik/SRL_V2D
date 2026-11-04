#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --job-name=P-rcem
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33/logs/rcem-%j.log
# usage: sbatch rollcem.sbatch BASE OUT SEED [ITER] [INIT_JSON]   (resumable: re-submit the same args)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ROLLCEM_COST=${ROLLCEM_COST:-worst}
if [ -z "${ROLLCEM_FIXED:-}" ]; then export ROLLCEM_FIXED="{}"; fi
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; P=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33
cd $R/code && timeout 2900 $R/.venv/bin/python -B -u $P/code/roll_cem.py --base $1 --output $2 --seed $3 --iterations ${4:-10} --samples 64 --workers 16 --time-budget 2500 ${5:+--init "$5"} ${6:+--std-scale $6}
