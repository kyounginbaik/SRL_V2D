#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=P-roll
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33/logs/roll-%A_%a.log
# usage: sbatch --array=0-N rollrefit.sbatch LISTFILE   (line: SCRIPT BASE OUTNAME args...; SCRIPT = roll_refit.py | wrist_refit.py)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CW_OUT=${CW_OUT:-staging_C}
P=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamP_ep33
line=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $1); set -- $line; S=$1; BASE=$2; OUT=$3; shift 3
if [ "$S" = T ]; then cd /tmp; exec /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $P/code/track_refit.py $BASE $P/$OUT "$@"; fi
if [ "$S" = X ]; then cd /tmp; exec /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $P/code/port_actor.py $BASE $P/$OUT; fi
cd /tmp; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u $P/code/$S $BASE $P/roll/$OUT "$@"
