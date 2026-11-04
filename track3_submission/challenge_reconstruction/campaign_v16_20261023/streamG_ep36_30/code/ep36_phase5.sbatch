#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --job-name=G36-p5
#SBATCH --array=0-9
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep36_p5-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep36; A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/code
ILC=$R/refinement_20261021_ep38/code/ilc_carry_generic.py; EPD=$R/refinement_20261021_ep36/episode_corrected/episode_000036
SR=$G/retention/seat17.5ilc_ret13.8_18.5_s0.5; RR=$G/retention/rm1_14.3ilc2_ret13.8_18.5_s0.5; BR=$G/retention/base_ret13.8_18.5_s0.75; O=$G/post_ret
cd /tmp
ilc() { $PY -B -u $ILC --source $1 --out $2 --side right --object wooden_spoon --t0 $3 --t1 18.6 --t-clear 18.9 --sigma-s 0.5 --episode-dir $EPD --iterations $4 --gain-rot $5 --gain-pos $6; }
rel() { $PY -B -u $A/release_variant.py --teacher $1 --episode-dir $EPD --out $2 --t-rel $3 --open-dur 0.3 --t-lift $4 --blend-dur 0.6; }
seat() { $PY -B -u $A/seated_grasp.py --teacher $1 --episode-dir $EPD --out $2 --t-seat $3 --t-on $4 --t-off 18.8 --ramp $5; }
case $SLURM_ARRAY_TASK_ID in
 0) rel $SR $O/sret_rel18.5_lift19.1 18.5 19.1;;
 1) rel $SR $O/sret_rel18.3_lift18.9 18.3 18.9;;
 2) ilc $SR $O/sret_ilc0.3 15.4 2 0.3 0.3;;
 3) ilc $SR $O/sret_ilc0.5 15.4 1 0.5 0.3;;
 4) ilc $RR $O/rret_ilc0.5 15.4 2 0.5 0.3;;
 5) ilc $RR $O/rret_ilc0.3 15.4 2 0.3 0.3;;
 6) rel $RR $O/rret_rel18.7_lift19.3 18.7 19.3;;
 7) seat $BR $O/bret_seat17.5_on15.4 17.5 15.4 0.5;;
 8) seat $BR $O/bret_seat17.0_on15.3 17.0 15.3 0.3;;
 9) ilc $BR $O/bret_ilc0.5 15.4 2 0.5 0.3;;
esac
