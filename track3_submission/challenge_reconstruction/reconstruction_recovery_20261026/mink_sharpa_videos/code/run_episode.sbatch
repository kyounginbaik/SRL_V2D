#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=mskv-ep
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/logs/mink_sharpa_video-%A_%a.log
# Usage: sbatch --array=1,3,5 run_episode.sbatch <bundle_dir>   (array index = episode number)
# Phases: decode (docker loader) -> map (venv) -> ik (docker mink 1.3) -> render (venv, EGL on node100).
set -euo pipefail
BUNDLE=${1:?bundle dir}
EP=${SLURM_ARRAY_TASK_ID:?use --array=<episode numbers>}
CODE=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/mink_sharpa_videos/code/mink_video.py
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MUJOCO_GL=egl
EXTRA=${MAP_ARGS:-}
echo "bundle=$BUNDLE episode=$EP host=$(hostname)"
python3 "$CODE" --bundle "$BUNDLE" --phase decode --episodes "$EP"
"$PY" -u "$CODE" --bundle "$BUNDLE" --phase map --episodes "$EP" $EXTRA
python3 "$CODE" --bundle "$BUNDLE" --phase ik --episodes "$EP"
"$PY" -u "$CODE" --bundle "$BUNDLE" --phase render --episodes "$EP" 2> >(grep -v 'libEGL warning' >&2)
echo DONE "$EP"
