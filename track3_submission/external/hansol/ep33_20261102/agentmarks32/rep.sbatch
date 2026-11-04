#!/bin/bash
#SBATCH -p debug
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=node200
#SBATCH -o /data1/hansol/ep33_20261102/logs/agentrep32_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; A=/data1/hansol/ep33_20261102/agentmarks32; AR=/data1/hansol/ep33_20261102/code/spoon_annot_roll2.py
declare -A B=([rp2]=$A/bundleK32RP2 [v14]=/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17)
WHO=${1:-user}
for k in rp2 v14; do
  M=$A/marks_ep32.json; [ $WHO = agent ] && M=$A/agent_marks.json
  $PY $AR $A/rep_${WHO}_$k 32 $M 0 258 --base ${B[$k]} --face-thr 0.25 --report-only > $A/rep_${WHO}_$k.txt 2>&1 &
done; wait; echo done
