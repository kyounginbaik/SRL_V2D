#!/bin/bash
# smooth the roll-0.8 correction (sigma 1.5 / 2.5 stored frames), then the same checks as check_roll.sh
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; C=/data1/hansol/mink_S1_20261031/code; W=/data1/hansol/ep33_20261102; R=$W/roll
V=/home/hansol/v2d/outputs/variants_20261030; S=$V/recon_E14_best_B2A_S1; e=33; E=episode_000033; CT=blue_cup
ANN=/home/hansol/v2d/outputs/ep32_robot_20261031/annot_json/ep33_f60_260.json; AR=/home/hansol/v2d/outputs/ep32_robot_20261031/code/spoon_annot_roll.py
L=""
for b in HGA HGB; do for sg in 15; do m=${b}R8s$sg; X=$R/recon_E14_best_B2A_S1_$m; mkdir -p $X
  $PY $W/code/ep33_smooth_corr.py $V/recon_E14_best_B2A_S1_$b/$E $R/recon_E14_best_B2A_S1_${b}R8/$E $X/$E --sigma ${sg:0:1}.${sg:1:1} > $R/smooth_$m.txt 2>&1
  L="$L $m=$X"
done; done
for kv in $L; do m=${kv%%=*}; X=${kv#*=}
  $PY /home/hansol/v2d/outputs/ep32_robot_20261031/code/smooth_check.py $e $ANN $S $X/$E $R/targets_$m > $R/check_$m.txt 2>&1 &
  $PY $C/contact_jitter.py $X $e $CT > $R/cj_$m.txt 2>&1 &
  $PY $AR $R/rep_$m $e $ANN 150 262 --base $X --face-thr 0.8 --report-only > $R/rep_$m.txt 2>&1 &
done; wait
$PY /data1/hansol/roll33_20261102/code/spoon_ball_roll.py $e $R/ballroll_s.json $L > $R/ballroll_s.txt 2>&1
