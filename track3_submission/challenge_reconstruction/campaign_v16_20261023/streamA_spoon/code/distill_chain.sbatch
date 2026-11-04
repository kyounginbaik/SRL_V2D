#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --job-name=A-distill
#SBATCH --array=0-3
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/logs/distill-%A_%a.log
# env: EP (episode number), TE (teacher dir), OUT (output base)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
C=("0.01 1024" "0.03 1024" "0.03 2048" "0.1 1024"); set -- ${C[$SLURM_ARRAY_TASK_ID]}
D=$OUT/distilled/fg$1_c$2
cd /tmp && $R/.venv/bin/python -B -u $R/code/distill_teacher.py --episode $EP --teacher $TE --output $D --feedback-gain $1 --centers $2
for K in margin all; do $R/.venv/bin/python -B -u $A/code/wrap_keys.py $EP $D $OUT/wrapped_$K/fg$1_c$2 --keys $K; done
