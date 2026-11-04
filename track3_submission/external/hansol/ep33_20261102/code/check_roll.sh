#!/bin/bash
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; C=/data1/hansol/mink_S1_20261031/code; W=/data1/hansol/ep33_20261102; R=$W/roll
V=/home/hansol/v2d/outputs/variants_20261030; S=$V/recon_E14_best_B2A_S1; e=33; E=episode_000033; CT=blue_cup; ANN=/home/hansol/v2d/outputs/ep32_robot_20261031/annot_json/ep33_f60_260.json
for m in HGAR HGAR8 HGBR8; do X=$R/recon_E14_best_B2A_S1_$m
  $PY /home/hansol/v2d/outputs/ep32_robot_20261031/code/smooth_check.py $e $ANN $S $X/$E $R/targets_$m > $R/check_$m.txt 2>&1 &
  $PY $C/contact_jitter.py $X $e $CT > $R/cj_$m.txt 2>&1 &
done
$PY /data1/hansol/roll33_20261102/code/spoon_ball_roll.py $e $R/ballroll.json HGA=$V/recon_E14_best_B2A_S1_HGA HGAR8=$R/recon_E14_best_B2A_S1_HGAR8 HGB=$V/recon_E14_best_B2A_S1_HGB HGBR8=$R/recon_E14_best_B2A_S1_HGBR8 > $R/ballroll.txt 2>&1 &
wait
