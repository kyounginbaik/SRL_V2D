#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --job-name=G36-p2
#SBATCH --array=0-19
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep36_p2-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
E36=$R/refinement_20261021_ep36; G=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep36
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/code
ILC=$R/refinement_20261021_ep38/code/ilc_carry_generic.py; T=$E36/search/roll_1_12.0_14.5; EPD=$E36/episode_corrected/episode_000036
cd /tmp; K=$SLURM_ARRAY_TASK_ID
ilc() { $PY -B -u $ILC --source $1 --out $2 --side right --object wooden_spoon --t0 $3 --t1 18.6 --t-clear 18.9 --sigma-s 0.5 --episode-dir $EPD --iterations $4 --gain-rot $5 --gain-pos $6; }
case $K in
 0) ilc $G/ilc/gr0.5_gp0.3/iter1 $G/ilc2/base_gr0.5_gp0.3_cont 15.4 2 0.5 0.3;;
 1) ilc $G/ilc/gr0.3_gp0.3/iter1 $G/ilc2/base_gr0.3_gp0.3_cont 15.4 2 0.3 0.3;;
 2) ilc $G/roll/roll_1_12.0_12.9_14.3_15.3 $G/ilc2/r1_14.3_gr0.5_gp0.3 15.4 3 0.5 0.3;;
 3) ilc $G/roll/roll_1_12.0_12.9_14.3_15.3 $G/ilc2/r1_14.3_gr0.3_gp0.3 15.4 3 0.3 0.3;;
 4) ilc $G/roll/roll_1_12.0_12.9_14.5_15.6 $G/ilc2/r1_14.5w_gr0.5_gp0.3 15.7 3 0.5 0.3;;
 5) ilc $G/roll/roll_1_12.0_12.9_14.5_15.6 $G/ilc2/r1_14.5w_gr0.3_gp0.3 15.7 3 0.3 0.3;;
 6) ilc $G/roll/roll_-1_12.0_12.9_14.5_15.6 $G/ilc2/rm1_14.5w_gr0.5_gp0.3 15.7 3 0.5 0.3;;
 7) ilc $G/roll/roll_-1_12.0_12.9_14.5_15.6 $G/ilc2/rm1_14.5w_gr0.3_gp0.3 15.7 3 0.3 0.3;;
 8) ilc $G/roll/roll_-1_12.0_12.9_14.3_15.3 $G/ilc2/rm1_14.3_gr0.5_gp0.3 15.4 3 0.5 0.3;;
 9) ilc $G/roll/roll_-1_12.0_12.9_14.3_15.3 $G/ilc2/rm1_14.3_gr0.3_gp0.3 15.4 3 0.3 0.3;;
 10) ilc $G/roll/roll_1_12.0_12.9_15.3_16.1 $G/ilc2/r1_15.3_gr0.5_gp0.3 16.3 3 0.5 0.3;;
 11) ilc $G/roll/roll_1_12.0_12.9_15.6_16.5 $G/ilc2/r1_15.6_gr0.5_gp0.3 16.7 3 0.5 0.3;;
 12) ilc $T $G/ilc2/base_wide_gr0.3_gp0.3 11.5 3 0.3 0.3;;
 13) ilc $T $G/ilc2/base_wide_gr0.5_gp0.3 11.5 3 0.5 0.3;;
 14) $PY -B -u $A/seated_grasp.py --teacher $G/roll/roll_1_12.0_12.9_14.3_15.3 --episode-dir $EPD --out $G/seat2/r1_14.3_seat17.5 --t-seat 17.5 --t-on 15.4 --t-off 18.8 --ramp 0.5;;
 15) $PY -B -u $A/seated_grasp.py --teacher $G/roll/roll_1_12.0_12.9_14.5_15.6 --episode-dir $EPD --out $G/seat2/r1_14.5w_seat17.5 --t-seat 17.5 --t-on 15.7 --t-off 18.8 --ramp 0.5;;
 16) $PY -B -u $A/seated_grasp.py --teacher $G/roll/roll_-1_12.0_12.9_14.5_15.6 --episode-dir $EPD --out $G/seat2/rm1_14.5w_seat17.5 --t-seat 17.5 --t-on 15.7 --t-off 18.8 --ramp 0.5;;
 17) $PY -B -u $A/seated_grasp.py --teacher $T --episode-dir $EPD --out $G/seat2/base_seat17.0_on15.3 --t-seat 17.0 --t-on 15.3 --t-off 18.8 --ramp 0.3;;
 18) $PY -B -u $A/seated_grasp.py --teacher $T --episode-dir $EPD --out $G/seat2/base_seat18.0_on15.4 --t-seat 18.0 --t-on 15.4 --t-off 18.8 --ramp 0.5;;
 19) ilc $G/seat/seat17.5_on15.4_ro0 $G/ilc2/seat17.5_gr0.5_gp0.3 15.4 3 0.5 0.3;;
esac
