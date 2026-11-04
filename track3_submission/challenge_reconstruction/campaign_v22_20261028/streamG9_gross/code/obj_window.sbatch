#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=12G -t 00:50:00 -J G9-obw -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/logs/obw_%a.log
J=(20:water_pitcher:c21 20:water_pitcher:c27 20:blue_cup:c27 8:beige_cup:c21 8:beige_cup:c27 1:white_pot_lid:c21 1:white_pot_lid:c27 32:blue_cup:c27)
x=${J[$SLURM_ARRAY_TASK_ID]}; IFS=: read e o t <<< "$x"
[ $t = c21 ] && TD=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target || TD=/data1/SRL_V2D/challenge_reconstruction/campaign_v23_20261028/target_compose27
cd /tmp && OMP_NUM_THREADS=2 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/code/spoon_window.py $e $t $TD /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/out/obj_window_ep${e}_${o}_$t.json $o 2>&1 | grep -v Warn
