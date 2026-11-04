#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --job-name=J-diag2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/logs/diag2-%j.log
source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamJ_ep3233/code/common.sh
cd /tmp; d=ep32/chainB/c2_R_9.0_11.0; $PY -B $J/code/diag_timeline.py $J/$d/execution/trajectory.npz $E32 $J/$d/execution/scene_physics.xml | tee $J/$d/timeline.txt
d=ep32/w57/rsc/c1_R_7.5_11.5; $PY -B $J/code/diag_timeline.py $J/$d/execution/trajectory.npz $E32R $J/$d/execution/scene_physics.xml | tee $J/$d/timeline.txt
