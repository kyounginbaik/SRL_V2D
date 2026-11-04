#!/bin/bash
# stronger facing (0.8) + silhouette IoU term in the two windows, balls x1 (ball detections there hit dark bowl-edge spots)
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; C=/home/hansol/v2d/outputs/ep32_robot_20261031/code; W=/data1/hansol/ep33_20261102; R=$W/roll
V=/home/hansol/v2d/outputs/variants_20261030; e=33; ANN=$C/../annot_json/ep33_f60_260.json
for b in HGA HGB; do
  $PY $C/spoon_annot_roll.py $R/${b}R8_w1 $e $ANN 150 200 --base $V/recon_E14_best_B2A_S1_$b --face-thr 0.8 --iou 160,192,1 > $R/${b}R8_w1.txt 2>&1
  $PY $C/spoon_annot_roll.py $R/${b}R8 $e $ANN 218 262 --base $R/${b}R8_w1/bundle --face-thr 0.8 --iou 226,254,1 > $R/${b}R8_w2.txt 2>&1
done
echo ROLL2-DONE
