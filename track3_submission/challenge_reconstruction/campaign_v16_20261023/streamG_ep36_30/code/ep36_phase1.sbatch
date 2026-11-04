#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:30:00
#SBATCH --job-name=G36-p1
#SBATCH --array=0-26
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/logs/ep36_p1-%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
PY=$R/.venv/bin/python
E36=$R/refinement_20261021_ep36
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep36
A=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamA_spoon/code
OLD=$R/refinement_20261018/episode_000036/physical_release_search_v2
T=$E36/search/roll_1_12.0_14.5
EPD=$E36/episode_corrected/episode_000036
cd /tmp
K=$SLURM_ARRAY_TASK_ID
ROLL=("1 12.0 12.9 13.5 14.3" "1 12.0 12.9 13.8 14.6" "1 12.0 12.9 14.0 14.8" "1 12.0 12.9 14.3 15.3" "1 12.0 12.9 14.5 15.6" "1 12.0 12.9 15.3 16.1" "1 12.0 12.9 15.6 16.5" "-1 12.0 12.9 13.5 14.3" "-1 12.0 12.9 13.8 14.6" "-1 12.0 12.9 14.0 14.8" "-1 12.0 12.9 14.3 15.3" "-1 12.0 12.9 14.5 15.6" "-1 12.0 12.9 15.3 16.1" "-1 12.0 12.9 15.6 16.5" "1 11.6 12.5 14.5 15.2" "1 12.0 13.2 14.5 15.2")
if [ $K -lt 16 ]; then set -- ${ROLL[$K]}; $PY -B -u $E36/code/roll_replan.py $OLD $G/roll/roll_$1_$2_$3_$4_$5 $1 $2 $3 $4 $5; exit; fi
K=$((K-16))
SEAT=("15.8 15.4 0" "16.5 15.4 0" "15.8 15.4 1" "16.5 15.4 1" "17.5 15.4 0")
if [ $K -lt 5 ]; then set -- ${SEAT[$K]}; RO=""; [ "$3" = "1" ] && RO="--rot-only"; $PY -B -u $A/seated_grasp.py --teacher $T --episode-dir $EPD --out $G/seat/seat$1_on$2_ro$3 --t-seat $1 --t-on $2 --t-off 18.8 --ramp 0.5 $RO; exit; fi
K=$((K-5))
ILC=("0.2 0.2" "0.3 0.3" "0.5 0.3" "0.3 0.0")
if [ $K -lt 4 ]; then set -- ${ILC[$K]}; $PY -B -u $R/refinement_20261021_ep38/code/ilc_carry_generic.py --source $T --out $G/ilc/gr$1_gp$2 --side right --object wooden_spoon --t0 15.4 --t1 18.6 --t-clear 18.9 --sigma-s 0.5 --episode-dir $EPD --iterations 2 --gain-rot $1 --gain-pos $2; exit; fi
K=$((K-4))
REL=("18.6 19.2" "18.8 19.4")
set -- ${REL[$K]}; $PY -B -u $A/release_variant.py --teacher $T --episode-dir $EPD --out $G/release/rel$1_d0.3_lift$2 --t-rel $1 --open-dur 0.3 --t-lift $2 --blend-dur 0.6
