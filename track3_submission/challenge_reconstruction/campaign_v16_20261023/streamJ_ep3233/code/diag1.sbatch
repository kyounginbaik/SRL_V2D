#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:20:00
#SBATCH --job-name=J-diag1
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/diag1-%j.log
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
cd /tmp
for d in ep33/p1/T1_relL15.6 ep33/p1/T2_relR14.4 ep33/p1/T1_relR14.4; do echo "### $d"; $PY -B $J/code/diag_timeline.py $J/$d/execution/trajectory.npz $E33 $J/$d/execution/scene_physics.xml | tee $J/$d/timeline.txt; done
for d in ep32/p1/T_relR10.9 ep32/p1/T_seat9.6_on8.6; do echo "### $d"; $PY -B $J/code/diag_timeline.py $J/$d/execution/trajectory.npz $E32 $J/$d/execution/scene_physics.xml | tee $J/$d/timeline.txt; done
