#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --job-name=ep38-distill
#SBATCH --array=0-7
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/logs/distill-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
T=("/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/ilc/gr0.8_gp0.5/iter3" "/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/ilc/gr0.6_gp0.6/iter3"); C=("0.1 1024" "0.03 1024" "0.01 1024" "0.1 2048")
i=$SLURM_ARRAY_TASK_ID; TE=${T[$((i/4))]}; set -- ${C[$((i%4))]}
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/distill_teacher.py --episode 38 --teacher $TE --output /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep38/distilled/$(basename $(dirname $TE))/fg$1_c$2 --feedback-gain $1 --centers $2
