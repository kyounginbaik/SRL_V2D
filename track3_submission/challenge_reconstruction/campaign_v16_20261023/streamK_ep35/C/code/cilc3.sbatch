#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35C-ilc3
#SBATCH --array=0-17
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/C/logs/cilc3-%A_%a.log
# round 3: continue position chains; roll on top. round 2: long-axis roll only (bodyx) / gentle position-only; full 6-DOF wrist ILC towards the compose-retargeted reference (episode dir C/episode/episode_000035)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35; EPD=$K/C/episode/episode_000035
T1=$K/teachers/ilc4/C1_none_gp0.3/iter0; T2=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/ep35/release/rel19.2_d0.3_lift19.8
# src name t0 t1 tclear axis gr gp sigma
A=$K/C/teachers/cilc2/relt_none_p0.3/iter2; B=$K/C/teachers/cilc2/v21t_none_p0.3/iter2; H=$K/C/teachers/cilc2/v21t_none_p0.3_hold/iter2
G=("$A A_p0.3 1.6 18.6 19.9 none 0.0 0.3 0.5" "$A A_p0.2 1.6 18.6 19.9 none 0.0 0.2 0.5" "$B B_p0.3 1.6 18.6 19.9 none 0.0 0.3 0.5" "$B B_p0.2 1.6 18.6 19.9 none 0.0 0.2 0.5" "$H H_p0.3 1.6 18.6 25.0 none 0.0 0.3 0.5" \
   "$A A_bx0.5_13.8 13.8 17.2 19.9 bodyx 0.5 0.0 0.3" "$A A_bx1.0_13.8 13.8 17.2 19.9 bodyx 1.0 0.0 0.3" "$A A_bx0.3_13.8 13.8 17.2 19.9 bodyx 0.3 0.0 0.3" "$B B_bx0.5_13.8 13.8 17.2 19.9 bodyx 0.5 0.0 0.3" "$B B_bx1.0_13.8 13.8 17.2 19.9 bodyx 1.0 0.0 0.3" \
   "$A A_bx0.3_1.6 1.6 18.6 19.9 bodyx 0.3 0.0 0.5" "$A A_z0.2 1.6 18.6 19.9 z 0.2 0.0 0.5" "$A A_all0.15 1.6 18.6 19.9 all 0.15 0.15 0.5" "$B B_all0.15 1.6 18.6 19.9 all 0.15 0.15 0.5" "$A A_all0.3_13.8 13.8 17.2 19.9 all 0.3 0.0 0.3" "$A A_bx0.5_p0.2_13.8 13.8 17.2 19.9 bodyx 0.5 0.2 0.3" \
   "$A A_bx0.5_1.6_s1 1.6 18.6 19.9 bodyx 0.5 0.0 1.0" "$H H_bx0.5_13.8 13.8 17.2 25.0 bodyx 0.5 0.0 0.3")
set -- ${G[$SLURM_ARRAY_TASK_ID]}
cd /tmp && $PY -B -u $K/code/ilc_axis.py --source $1 --out $K/C/teachers/cilc3/$2 --side right --object wooden_spoon --t0 $3 --t1 $4 --t-clear $5 --sigma-s $9 --episode-dir $EPD --iterations 3 --gain-rot $7 --gain-pos $8 --rot-axis $6
