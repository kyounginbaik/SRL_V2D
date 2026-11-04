#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=H1-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/logs/stage-v17actor_s123_fg0.01_c1024-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/code/stage_ep1_component.py --episode 1 --distilled /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/distilled/v17actor_s123_fg0.01_c1024 --base /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/staging/stage_v17actor_s123_fg0.01_c1024 --teacher /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep1/lid/shift_0.02_0_0_12.3 --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamH_ep1/README.md
