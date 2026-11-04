#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=16G -t 00:50:00 -J G9-pf -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross/logs/pf_%A_%a.log
G=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG9_gross
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $G/code/$JOBS); set -- $LINE
mkdir -p $(dirname $G/out/pf/$4)
cd /tmp && OMP_NUM_THREADS=2 $PY -B $G/code/perframe.py $1 $2 $3 $G/out/pf/$4.npz $5 2>&1 | grep -v Warn
