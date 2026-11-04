#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35-ilc
#SBATCH --array=0-21
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/logs/ilc1-%A_%a.log
# axis-restricted ILC (code/ilc_axis.py) on the Stream A release-fixed teacher rel19.2_d0.3_lift19.8 (2.05 / 19.0)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35; EPD=$R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter
T=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/ep35/release/rel19.2_d0.3_lift19.8
# t0 t1 tclear rot_axis gain_rot gain_pos
G=("6.5 18.6 19.9 z 0.5 0.5" "6.5 18.6 19.9 z 1.0 1.0" "6.5 18.6 19.9 z 0.8 0.8" "6.5 18.6 19.9 z 0.0 1.0" "6.5 18.6 19.9 z 1.0 0.0" "6.5 18.6 19.9 z 0.5 1.0" \
   "9.0 18.6 19.9 z 0.5 0.5" "9.0 18.6 19.9 z 1.0 1.0" "9.0 18.6 19.9 z 0.8 0.8" "9.0 18.6 19.9 z 0.0 1.0" "9.0 18.6 19.9 z 1.0 0.0" \
   "4.5 18.6 19.9 z 0.5 0.5" "4.5 18.6 19.9 z 1.0 1.0" "4.5 18.6 19.9 z 0.8 0.8" "4.5 18.6 19.9 z 0.0 1.0" \
   "9.0 18.6 19.9 all 0.3 0.3" "9.0 18.6 19.9 all 0.3 0.0" "9.0 18.6 19.9 none 0.0 0.5" "6.5 18.6 19.9 none 0.0 0.5" "13.0 18.6 19.9 z 1.0 1.0" "13.0 18.6 19.9 z 0.5 0.5" "6.5 18.6 19.9 z 0.3 0.3")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -B -u $K/code/ilc_axis.py --source $T --out $K/teachers/ilc1/w$1_$2_$4_gr$5_gp$6 --side right --object wooden_spoon --t0 $1 --t1 $2 --t-clear $3 --sigma-s 0.5 --episode-dir $EPD --iterations 2 --gain-rot $5 --gain-pos $6 --rot-axis $4
