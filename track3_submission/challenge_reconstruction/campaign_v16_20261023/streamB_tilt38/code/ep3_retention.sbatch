#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --job-name=sB-ep3-ret
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/logs/ep3-ret-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${SOURCE:?}" "${OUT:?}"
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/refine_versioned_retention_full.py --source "$SOURCE" --episode-dir /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamB_tilt38/ep3/calibrated_inputs/episodes/episode_000003 --output "$OUT" --start-s ${START:-4.5} --stop-s ${STOP:-14.45} --side ${SIDE:-right} --protect-object ${PROTECT:-white_pot} --samples 96 --iterations ${ITERS:-14} --workers 16 --seed ${SEED:-71}
