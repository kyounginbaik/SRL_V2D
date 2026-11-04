#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35-ilc4
#SBATCH --array=0-13
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/logs/ilc4-%A_%a.log
# round 4: continue the rotation-friendly chains of ilc2
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35; EPD=$R/refinement_20261018/wrist_chart/episode_000035/chart_0/adapter
T=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/ep35/release/rel19.2_d0.3_lift19.8
C1=$K/teachers/ilc2/C_none_gp1.0/iter1; CZ=$K/teachers/ilc2/C_z_gr0.2_gp0.3/iter1; F=$K/teachers/ilc2/F_w6.5_none_gp0.7/iter1; B4=$K/teachers/ilc2/B_none_gp0.5/iter0; B3=$K/teachers/ilc2/B_none_gp0.3/iter0
# src out t0 axis gr gp sigma
G=("$C1 C1_none_gp0.3 6.5 none 0.0 0.3 0.5" "$C1 C1_none_gp0.5 6.5 none 0.0 0.5 0.5" "$C1 C1_z_gr0.2_gp0.3 6.5 z 0.2 0.3 0.5" "$C1 C1_z_gr0.1_gp0.3 6.5 z 0.1 0.3 0.5" \
   "$CZ CZ_none_gp0.5 6.5 none 0.0 0.5 0.5" "$CZ CZ_z_gr0.2_gp0.3 6.5 z 0.2 0.3 0.5" "$CZ CZ_none_gp0.3 6.5 none 0.0 0.3 0.5" \
   "$F F_none_gp0.3 6.5 none 0.0 0.3 0.5" "$F F_z_gr0.15_gp0.3 6.5 z 0.15 0.3 0.5" "$B4 B4_none_gp0.3 6.5 none 0.0 0.3 0.5" "$B3 B3_none_gp0.3 6.5 none 0.0 0.3 0.5" "$B3 B3_z_gr0.15_gp0.3 6.5 z 0.15 0.3 0.5" \
   "$T F2_w6.5_none_gp0.7_s0.8 6.5 none 0.0 0.7 0.8" "$T F2_w6.5_none_gp1.0_s0.8 6.5 none 0.0 1.0 0.8")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -B -u $K/code/ilc_axis.py --source $1 --out $K/teachers/ilc4/$2 --side right --object wooden_spoon --t0 $3 --t1 18.6 --t-clear 19.9 --sigma-s $7 --episode-dir $EPD --iterations 2 --gain-rot $5 --gain-pos $6 --rot-axis $4
