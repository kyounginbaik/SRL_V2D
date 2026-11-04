#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35-ilc3
#SBATCH --array=0-15
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/logs/ilc3-%A_%a.log
# round 3: late-window yaw-only corrections (just before the placement) on top of the position-corrected teachers
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35; EPD=$R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter
A=$K/teachers/ilc1/w9.0_18.6_none_gr0.0_gp0.5/iter1; B2=$K/teachers/ilc2/B_none_gp0.5/iter0
# src out t0 axis gr gp
G=("$A A_lateyaw17.0_gr0.5 17.0 z 0.5 0.0" "$A A_lateyaw17.0_gr1.0 17.0 z 1.0 0.0" "$A A_lateyaw16.0_gr0.5 16.0 z 0.5 0.0" "$A A_lateyaw16.0_gr1.0 16.0 z 1.0 0.0" "$A A_lateyaw17.5_gr1.0 17.5 z 1.0 0.0" "$A A_lateyaw18.0_gr1.0 18.0 z 1.0 0.0" "$A A_lateyaw17.0_gr0.5_gp0.3 17.0 z 0.5 0.3" "$A A_lateyaw16.0_gr0.3 16.0 z 0.3 0.0" \
   "$B2 B2_lateyaw17.0_gr0.5 17.0 z 0.5 0.0" "$B2 B2_lateyaw17.0_gr1.0 17.0 z 1.0 0.0" "$B2 B2_lateyaw16.0_gr0.5 16.0 z 0.5 0.0" "$B2 B2_lateyaw16.0_gr1.0 16.0 z 1.0 0.0" "$B2 B2_lateyaw17.5_gr1.0 17.5 z 1.0 0.0" "$B2 B2_lateyaw18.0_gr1.0 18.0 z 1.0 0.0" "$B2 B2_lateyaw14.0_gr0.3 14.0 z 0.3 0.0" "$B2 B2_lateyaw12.0_gr0.3 12.0 z 0.3 0.0")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -B -u $K/code/ilc_axis.py --source $1 --out $K/teachers/ilc3/$2 --side right --object wooden_spoon --t0 $3 --t1 18.6 --t-clear 19.9 --sigma-s 0.3 --episode-dir $EPD --iterations 2 --gain-rot $5 --gain-pos $6 --rot-axis $4
