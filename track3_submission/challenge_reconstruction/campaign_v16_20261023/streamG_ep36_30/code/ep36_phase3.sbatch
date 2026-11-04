#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --job-name=G36-p3
#SBATCH --array=0-17
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep36_p3-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
E36=$R/refinement_20261021_ep36; G=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep36
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/code
ILC=$R/refinement_20261021_ep38/code/ilc_carry_generic.py; T=$E36/search/roll_1_12.0_14.5; EPD=$E36/episode_corrected/episode_000036
cd /tmp; K=$SLURM_ARRAY_TASK_ID
ilc() { $PY -B -u $ILC --source $1 --out $2 --side right --object wooden_spoon --t0 $3 --t1 18.6 --t-clear 18.9 --sigma-s 0.5 --episode-dir $EPD --iterations $4 --gain-rot $5 --gain-pos $6; }
rel() { $PY -B -u $A/release_variant.py --teacher $1 --episode-dir $EPD --out $2 --t-rel $3 --open-dur 0.3 --t-lift $4 --blend-dur 0.6; }
seat() { $PY -B -u $A/seated_grasp.py --teacher $1 --episode-dir $EPD --out $2 --t-seat $3 --t-on $4 --t-off 18.8 --ramp $5; }
case $K in
 0) ilc $G/ilc2/rm1_14.3_gr0.5_gp0.3/iter2 $G/ilc3/rm1_14.3_gr0.5_cont 15.4 2 0.5 0.3;;
 1) ilc $G/ilc2/rm1_14.5w_gr0.3_gp0.3/iter2 $G/ilc3/rm1_14.5w_gr0.3_cont 15.7 2 0.3 0.3;;
 2) ilc $G/ilc2/base_gr0.5_gp0.3_cont/iter1 $G/ilc3/base_gr0.5_cont2_g0.3 15.4 2 0.3 0.3;;
 3) ilc $G/ilc2/r1_14.5w_gr0.5_gp0.3/iter2 $G/ilc3/r1_14.5w_gr0.5_cont 15.7 2 0.5 0.3;;
 4) ilc $G/seat2/base_seat17.0_on15.3 $G/ilc3/seat17.0_gr0.5_gp0.3 15.3 1 0.5 0.3;;
 5) ilc $G/seat2/base_seat17.0_on15.3 $G/ilc3/seat17.0_gr0.3_gp0.3 15.3 2 0.3 0.3;;
 6) ilc $G/seat/seat17.5_on15.4_ro0 $G/ilc3/seat17.5_gr0.3_gp0.3 15.4 2 0.3 0.3;;
 7) ilc $G/seat/seat17.5_on15.4_ro0 $G/ilc3/seat17.5_gr0.5_gp0.0 15.4 1 0.5 0.0;;
 8) ilc $G/seat/seat17.5_on15.4_ro0 $G/ilc3/seat17.5_gr0.3_gp0.15 15.4 1 0.3 0.15;;
 9) rel $G/ilc2/seat17.5_gr0.5_gp0.3/iter0 $G/release3/seat17.5ilc_rel18.5_lift19.1 18.5 19.1;;
 10) rel $G/ilc2/seat17.5_gr0.5_gp0.3/iter0 $G/release3/seat17.5ilc_rel18.7_lift19.3 18.7 19.3;;
 11) rel $G/seat2/base_seat17.0_on15.3 $G/release3/seat17.0_rel18.5_lift19.1 18.5 19.1;;
 12) rel $G/seat2/base_seat17.0_on15.3 $G/release3/seat17.0_rel18.7_lift19.3 18.7 19.3;;
 13) rel $G/ilc2/rm1_14.3_gr0.5_gp0.3/iter2 $G/release3/rm1_14.3ilc_rel18.7_lift19.3 18.7 19.3;;
 14) seat $G/roll/roll_-1_12.0_12.9_14.3_15.3 $G/seat3/rm1_14.3_seat17.0_on15.3 17.0 15.3 0.3;;
 15) seat $T $G/seat3/base_seat16.8_on15.3 16.8 15.3 0.3;;
 16) seat $T $G/seat3/base_seat17.2_on15.2 17.2 15.2 0.3;;
 17) seat $G/ilc/gr0.5_gp0.3/iter1 $G/seat3/ilc1_seat17.0_on15.3 17.0 15.3 0.3;;
esac
