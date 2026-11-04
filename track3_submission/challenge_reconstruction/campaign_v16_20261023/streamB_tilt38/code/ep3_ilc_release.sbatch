#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=sB-ep3-ilc
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/logs/ep3-ilc-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SRC:?}" "${OUT:?}"
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/ilc_reference.py --source "$SRC" --out "$OUT" --iterations ${ITERS:-4} --gain-rot ${GR:-0.6} --gain-pos ${GP:-0.6} --press-m 0.01 --open-s 0.8 --retreat-m 0.08 --retreat-s 0.6 --no-blend ${EXTRA:-}
