#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=H1-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/logs/stage-prev_s128_fg0.003_c512-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/code/stage_ep1_component.py --episode 1 --distilled /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/distilled/prev_s128_fg0.003_c512 --base /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/staging/stage_prev_s128_fg0.003_c512 --teacher /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep1/lid/shift_0.02_0_0_12.8 --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/README.md
