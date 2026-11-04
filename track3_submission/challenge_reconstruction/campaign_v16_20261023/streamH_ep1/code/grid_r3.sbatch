#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:50:00
#SBATCH --job-name=H1-grid
#SBATCH --array=0-2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/logs/grid-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
H=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1
T=(r3_11.8_15.5 r3_12.3_16.7 r3_12.8_16.7); N=${T[$SLURM_ARRAY_TASK_ID]}
cd /tmp
$PY -u $H/code/actor_grid_h.py --episode 1 --teacher /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep1/lid/$N --output $H/distilled/$N --episode-dir $H/staging/episode_corrected/episode_000001 --workers 8 --seed 44 --grid 512,0.003,0.0001 1024,0.003,0.0001 2048,0.003,0.0001 512,0.01,0.0005 1024,0.01,0.0005 2048,0.01,0.0005 1024,0.03,0.0005 2048,0.001,0.00005
