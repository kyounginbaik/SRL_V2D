#!/bin/bash
#SBATCH -p debug --nodelist=node100 -c 4 --mem=24G -t 00:50:00 -J C30-psc -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/logs/pkgscore_%j.log
# usage: sbatch pkgscore.sbatch PKG TARGET OUT.json [PKG TARGET OUT.json ...]
cd /tmp; export OMP_NUM_THREADS=1
while [ $# -ge 3 ]; do /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/code/pkgscore.py $1 $2 $3; shift 3; done
