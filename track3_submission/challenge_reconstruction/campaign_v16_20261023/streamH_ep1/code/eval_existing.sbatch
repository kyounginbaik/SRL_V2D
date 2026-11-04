#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --job-name=H1-evalx
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/logs/evalx-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
D=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep1/distill
cd /tmp
$PY -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/code/eval_existing.py v17actor_s123_fg0.01_c1024 $D/shift_0.02_0_0_12.3/fg0.01_c1024/training
$PY -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/code/eval_existing.py prev_s128_fg0.003_c512 $D/shift_0.02_0_0_12.8/fg0.003_c512/training
