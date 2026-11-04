#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=m13
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/logs/margins_v13-%j.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
: "${EP:?}" "${ACTOR:?}" "${OUT:?}"
cd /tmp; /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep32/code/search_finger_margins_v13.py --episode $EP --actor $ACTOR --output $OUT --tolerance-deg 5 --spoon-weight ${SW:-1.0} --iterations 16 --samples 96 --workers 16 ${EXTRA:-}
