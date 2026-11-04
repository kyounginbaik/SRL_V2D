#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --job-name=A35-mg
#SBATCH --array=0-9
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/logs/mg35-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
G=("thumb_IP=0.15" "thumb_IP=0.3" "index_MCP_FE=0.1" "index_MCP_FE=0.2" "index_DIP=0.3" "thumb_IP=0.15,index_MCP_FE=0.1" "thumb_IP=0.3,index_MCP_FE=0.2" "thumb_IP=0.15,index_DIP=0.3" "thumb_MCP_FE=0.2" "thumb_CMC_FE=0.15,index_MCP_FE=0.1")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $R/.venv/bin/python -B -u $A/code/curl_variant.py --teacher $A/ep35/release/rel19.2_d0.3_lift19.8 --episode-dir $R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter --out $A/ep35/margin/$1 --fingers "" --add "$1" --t-on 1.6 --t-off 18.9 --ramp 0.4
