#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=sB-ep3-rel
#SBATCH --array=0-7
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/logs/ep3-rel-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SOURCE:?}" "${SPEC:?}" "${OUTROOT:?}"
combos=("1.5 0.6 handle 1.2" "2.5 0.6 handle 0.6" "2.5 0.6 handle 1.2" "1.5 0.65 handle 0.6" "1.5 0.6 handle_up 0.6" "2.5 0.65 handle 1.2" "1.5 0.6 handle 0.6 0.04" "2.5 0.6 handle_up 1.2")
set -- ${combos[$SLURM_ARRAY_TASK_ID]}
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/rewrite_release.py --source "$SOURCE" --spec "$SPEC" --out "$OUTROOT/open$1_frac$2_$3_rs$4_rm${5:-0.08}" --open-s $1 --open-frac $2 --retreat-dir $3 --retreat-s ${4:-0.6} --retreat-m ${5:-0.08}
