#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:30:00
#SBATCH --job-name=A35-curl
#SBATCH --array=0-13
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/logs/curl35-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
G=("middle,ring 0.5 0 0" "middle,ring 0.7 0 0" "middle,ring 0.9 0 0" "middle,ring,pinky 0.5 0 0" "middle,ring,pinky 0.7 0 0" "middle,ring,pinky 0.9 0 0" "middle,ring 0.7 0.4 0" "middle,ring,pinky 0.7 0.4 0" "middle,ring,pinky 0.9 0.4 0" "middle,ring,pinky 0.7 0 0.3" "middle,ring,pinky 0.9 0.4 0.3" "middle 0.7 0 0" "middle,ring,pinky 1.0 0.5 0.3" "none 0 0 0")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
F=$1; [ "$F" = none ] && F=""
cd /tmp && $R/.venv/bin/python -B -u $A/code/curl_variant.py --teacher $R/refinement_20261021_ep35/search/ret_12.5_18.4_s0.5 --episode-dir $R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter --out $A/ep35/curl/f${1}_c$2_ip$3_tip$4 --fingers "$F" --curl $2 --index-pip $3 --thumb-ip $4 --t-on 1.5 --t-off 18.8 --only-close
