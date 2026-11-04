#!/bin/bash
#SBATCH -p debug --nodelist=node01 -c 2 --mem=16G -t 00:50:00 -J G14-sc -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG14/logs/sc_%A_%a.log
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /data1/SRL_V2D
LIST=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG14/code/${LIST:-score_list1.txt}; STRIDE=${STRIDE:-4}
for i in $(seq 0 $((STRIDE-1))); do
  LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID*STRIDE+i+1))p" "$LIST"); [ -z "$LINE" ] && continue
  /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -B tools/reconstruction_audit/score_bundle.py "$LINE" --metrics > /dev/null || echo "FAIL $LINE"
  echo "DONE $LINE"
done
