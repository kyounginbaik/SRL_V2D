#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --job-name=ep19-distill
#SBATCH --array=0-11
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep19/logs/distill-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
T=(/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep19/ilc/gr0.6_gp0.6/iter2 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep19/ilc/gr0.4_gp0.4/iter3 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep19/ilc/gr0.8_gp0.8/iter2); C=("0.1 1024" "0.03 1024" "0.1 2048" "0.3 1024")
i=$SLURM_ARRAY_TASK_ID; TE=${T[$((i/4))]}; set -- ${C[$((i%4))]}; N=$(basename $(dirname $TE))_$(basename $TE)
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/distill_teacher.py --episode 19 --teacher $TE --output /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep19/distilled/$N/fg$1_c$2 --feedback-gain $1 --centers $2 && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep19/code/wrap_donor_keys.py 19 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep19/distilled/$N/fg$1_c$2 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep19/wrapped/$N/fg$1_c$2 && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep19/code/wrap_donor_keys.py 19 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep19/distilled/$N/fg$1_c$2 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep19/wrapped_zero/$N/fg$1_c$2 --zero-margins
