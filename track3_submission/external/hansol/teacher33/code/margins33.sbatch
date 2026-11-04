#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 12
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH -J T33-margins
#SBATCH -o /data1/hansol/teacher33/logs/%x-%j.log
# usage: margins33.sbatch ACTOR_DIR OUT [SEED] [SAMPLES] [ITERS]  -- team finger-margin CEM (streamK search_margins_k.py, as retarget32 margins32c.sbatch);
# the actor must have been evaluated in the snapped world worlds/R33c18_W1s (distill33.sbatch does that)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/hansol/teacher33/tmp PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
cd /data1/hansol/teacher33/tmp; /home/hansol/v2d/.venvs/v2d-host/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/code/search_margins_k.py --episode 33 --actor $1 --output $2 --samples ${4:-96} --iterations ${5:-12} --workers 12 --seed ${3:-3331} 2>&1 | grep -v -E "Warn|warn\("
