#!/bin/bash
#SBATCH --job-name=dexagent-float
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-4%5
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/logs/test_%A_%a.log
set -euo pipefail
TASK_ROOT=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data1/SRL_V2D/third_party/DexAgent:$TASK_ROOT/code
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1 EGL_PLATFORM=surfaceless
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
EPISODES=(1 3 5 8 10)
"$TASK_ROOT/.venv/bin/python" -u "$TASK_ROOT/code/run_episode_test.py" \
  --episode "${EPISODES[$SLURM_ARRAY_TASK_ID]}" --seed 44 --workers 4 \
  --samples 16 --max-minutes 10 --tag "${DEX_TEST_TAG:-pilot}"
