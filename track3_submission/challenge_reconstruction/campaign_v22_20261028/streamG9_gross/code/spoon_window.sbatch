#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=12G -t 00:50:00 -J G9-spw -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/logs/spw_%a.log
J=(3:c21 3:c27 32:c21 32:c27 33:c21 33:c27 35:c21 35:c27 36:c21 36:c27 5:c27 6:c27 16:c27)
x=${J[$SLURM_ARRAY_TASK_ID]}; e=${x%%:*}; t=${x##*:}
[ $t = c21 ] && TD=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target || TD=/data1/SRL_V2D/challenge_reconstruction/campaign_v23_20261028/target_compose27
cd /tmp && OMP_NUM_THREADS=2 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/code/spoon_window.py $e $t $TD /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/out/spoon_window_ep${e}_$t.json 2>&1 | grep -v Warn
