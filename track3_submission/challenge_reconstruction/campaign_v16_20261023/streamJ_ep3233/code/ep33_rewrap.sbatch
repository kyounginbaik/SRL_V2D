#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --job-name=J33-rewrap
#SBATCH --array=0-6
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/ep33_rewrap-%A_%a.log
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
BASE=$R/refinement_20261021_ep33/v13/m2_Econt; G=$J/ep33/rewrap; cd /tmp; K=$SLURM_ARRAY_TASK_ID
rw() { o=$1; shift; $PY -B -u $J/code/rewrap33.py $BASE $G/$o "$@"; }
case $K in
 0) rw start16.5_open0.6 release0_start_s=16.5 release0_open_duration_s=0.6;;
 1) rw start16.8_open0.6 release0_start_s=16.8 release0_open_duration_s=0.6;;
 2) rw start17.0_open0.4 release0_start_s=17.0 release0_open_duration_s=0.4;;
 3) rw start16.2_open1.0 release0_start_s=16.2 release0_open_duration_s=1.0;;
 4) rw start16.2_open0.6_retreat2.0 release0_start_s=16.2 release0_open_duration_s=0.6 release0_retreat_duration_s=2.0;;
 5) rw start16.5_open0.6_delta1.5 release0_start_s=16.5 release0_open_duration_s=0.6 release0_open_delta_rad=x1.5;;
 6) rw start16.2_open0.6_retreatz0.03 release0_start_s=16.2 release0_open_duration_s=0.6 release0_retreat_m=x0.4;;
esac
