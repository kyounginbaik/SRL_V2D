#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=12G -t 00:50:00 -J G9-scant -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/logs/scant_%a.log
EPS=(1 3 5 6 8 10 14 15 16 19 20 25 28 30 32 33 35 36 37 38)
cd /tmp && OMP_NUM_THREADS=2 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/code/build_scan_target.py /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/target_final_v6 /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/scorer_only_scan_target_v6 /data1/SRL_V2D/challenge_reconstruction/campaign_v23_20261028/target_compose27 ${EPS[$SLURM_ARRAY_TASK_ID]}
