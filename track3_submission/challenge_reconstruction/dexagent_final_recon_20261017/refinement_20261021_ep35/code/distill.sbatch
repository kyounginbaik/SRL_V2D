#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --job-name=ep35-distill
#SBATCH --array=0-9
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/logs/distill-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
T=(/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/search/ret_12.5_18.4_s0.5 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/search/ret_14.0_18.4_s0.75); C=("0.03 1024" "0.1 1024" "0.01 1024" "0.03 2048" "0.3 1024")
i=$SLURM_ARRAY_TASK_ID; TE=${T[$((i/5))]}; set -- ${C[$((i%5))]}; N=$(basename $TE)
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/distill_teacher.py --episode 35 --teacher $TE --output /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/distilled/$N/fg$1_c$2 --feedback-gain $1 --centers $2
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep19/code/wrap_donor_keys.py 35 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/distilled/$N/fg$1_c$2 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/wrapped/$N/fg$1_c$2
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep19/code/wrap_donor_keys.py 35 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/distilled/$N/fg$1_c$2 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep35/wrapped_zero/$N/fg$1_c$2 --zero-margins
