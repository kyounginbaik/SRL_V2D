#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35-ilc2
#SBATCH --array=0-19
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/logs/ilc2-%A_%a.log
# round 2: continue the position-only chains of ilc1 and a few fresh variants (code/ilc_axis.py)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35; EPD=$R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter
T=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/ep35/release/rel19.2_d0.3_lift19.8
A=$K/teachers/ilc1/w9.0_18.6_none_gr0.0_gp0.5/iter1; B=$K/teachers/ilc1/w6.5_18.6_none_gr0.0_gp0.5/iter1; C=$K/teachers/ilc1/w6.5_18.6_z_gr0.0_gp1.0/iter1; D=$K/teachers/ilc1/w4.5_18.6_z_gr0.0_gp1.0/iter0
# src out t0 axis gr gp sigma
G=("$A A_none_gp0.5 9.0 none 0.0 0.5 0.5" "$A A_none_gp0.3 9.0 none 0.0 0.3 0.5" "$A A_z_gr0.2_gp0.5 9.0 z 0.2 0.5 0.5" "$A A_z_gr0.3_gp0.3 9.0 z 0.3 0.3 0.5" \
   "$B B_none_gp0.5 6.5 none 0.0 0.5 0.5" "$B B_none_gp0.3 6.5 none 0.0 0.3 0.5" "$B B_z_gr0.2_gp0.5 6.5 z 0.2 0.5 0.5" \
   "$C C_none_gp0.5 6.5 none 0.0 0.5 0.5" "$C C_none_gp0.3 6.5 none 0.0 0.3 0.5" "$C C_none_gp1.0 6.5 none 0.0 1.0 0.5" "$C C_z_gr0.2_gp0.3 6.5 z 0.2 0.3 0.5" \
   "$D D_none_gp0.5 4.5 none 0.0 0.5 0.5" \
   "$T F_w9.0_none_gp0.7 9.0 none 0.0 0.7 0.5" "$T F_w6.5_none_gp0.7 6.5 none 0.0 0.7 0.5" "$T F_w7.5_none_gp0.5 7.5 none 0.0 0.5 0.5" "$T F_w9.0_none_gp0.5_s0.8 9.0 none 0.0 0.5 0.8" "$T F_w9.0_none_gp0.5_s0.3 9.0 none 0.0 0.5 0.3" "$T F_w9.0_none_gp0.3 9.0 none 0.0 0.3 0.5" "$T F_w5.5_none_gp0.5 5.5 none 0.0 0.5 0.5" "$T F_w9.0_z_gr0.2_gp0.5 9.0 z 0.2 0.5 0.5")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -B -u $K/code/ilc_axis.py --source $1 --out $K/teachers/ilc2/$2 --side right --object wooden_spoon --t0 $3 --t1 18.6 --t-clear 19.9 --sigma-s $7 --episode-dir $EPD --iterations 2 --gain-rot $5 --gain-pos $6 --rot-axis $4
