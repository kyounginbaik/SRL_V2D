#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35C-ilc
#SBATCH --array=0-17
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/C/logs/cilc1-%A_%a.log
# full 6-DOF wrist ILC towards the compose-retargeted reference (episode dir C/episode/episode_000035)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35; EPD=$K/C/episode/episode_000035
T1=$K/teachers/ilc4/C1_none_gp0.3/iter0; T2=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/ep35/release/rel19.2_d0.3_lift19.8
# src name t0 t1 tclear axis gr gp sigma
G=("$T1 v21t_all_g1.0_s0.5 1.6 18.6 19.9 all 1.0 1.0 0.5" "$T1 v21t_all_g0.7_s0.5 1.6 18.6 19.9 all 0.7 0.7 0.5" "$T1 v21t_all_g0.5_s0.5 1.6 18.6 19.9 all 0.5 0.5 0.5" "$T1 v21t_all_g1.0_s1.0 1.6 18.6 19.9 all 1.0 1.0 1.0" "$T1 v21t_all_g0.7_s1.0 1.6 18.6 19.9 all 0.7 0.7 1.0" \
   "$T1 v21t_all_r1.0p0.5_s0.5 1.6 18.6 19.9 all 1.0 0.5 0.5" "$T1 v21t_all_r0.5p1.0_s0.5 1.6 18.6 19.9 all 0.5 1.0 0.5" "$T1 v21t_none_p1.0_s0.5 1.6 18.6 19.9 none 0.0 1.0 0.5" "$T1 v21t_all_g1.0_s0.5_w3 3.0 18.6 19.9 all 1.0 1.0 0.5" \
   "$T1 v21t_all_g1.0_s0.5_hold 1.6 18.6 25.0 all 1.0 1.0 0.5" "$T1 v21t_all_g0.7_s1.0_hold 1.6 18.6 25.0 all 0.7 0.7 1.0" "$T1 v21t_all_g1.0_s0.5_t17.5 1.6 17.5 19.9 all 1.0 1.0 0.5" \
   "$T2 relt_all_g1.0_s0.5 1.6 18.6 19.9 all 1.0 1.0 0.5" "$T2 relt_all_g0.7_s0.5 1.6 18.6 19.9 all 0.7 0.7 0.5" "$T2 relt_all_g0.7_s1.0 1.6 18.6 19.9 all 0.7 0.7 1.0" "$T2 relt_all_g0.5_s0.5 1.6 18.6 19.9 all 0.5 0.5 0.5" "$T2 relt_all_r1.0p0.5_s0.5 1.6 18.6 19.9 all 1.0 0.5 0.5" "$T2 relt_all_g1.0_s0.5_hold 1.6 18.6 25.0 all 1.0 1.0 0.5")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -B -u $K/code/ilc_axis.py --source $1 --out $K/C/teachers/cilc1/$2 --side right --object wooden_spoon --t0 $3 --t1 $4 --t-clear $5 --sigma-s $9 --episode-dir $EPD --iterations 3 --gain-rot $7 --gain-pos $8 --rot-axis $6
