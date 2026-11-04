#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=12G -t 00:50:00 -J G9-adopt -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/logs/adopt_%a.log
EPS=(3 5 6 15 16 32 33 35 36 37)
cd /tmp && OMP_NUM_THREADS=2 /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/code/g9_spoon_adopt.py /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/bundle_tmp/ep${EPS[$SLURM_ARRAY_TASK_ID]} ${EPS[$SLURM_ARRAY_TASK_ID]} 2>&1 | grep -v Warn
