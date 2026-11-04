#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:30:00
#SBATCH --job-name=J33-pkbase
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/pickup33_base-%j.log
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
set -e; cd /tmp; T2=$R/refinement_20261021_ep33/search/tail/c1_12.4_16.5; B=$J/ep33/pickup2/base_T2_relR14.4_lift17.3
rel $T2 $E33 $B right wooden_spoon 14.4 17.3
$PY -B $J/code/prepare_pickup33.py $B $J/ep33/pickup2/specs 14.4 3.5,4.5,3.0
$PY -B $J/code/diag_timeline.py $B/execution/trajectory.npz $E33 $B/execution/scene_physics.xml | tee $B/timeline.txt
