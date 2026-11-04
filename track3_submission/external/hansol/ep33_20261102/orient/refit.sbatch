#!/bin/bash
#SBATCH -p debug
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /data1/hansol/ep33_20261102/logs/orientfit_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102; O=$W/orient/fit; V=/home/hansol/v2d/outputs/variants_20261030; E=episode_000033
AR=/home/hansol/v2d/outputs/ep32_robot_20261031/code/spoon_annot_roll.py; ANN=/home/hansol/v2d/outputs/ep32_robot_20261031/annot_json/ep33_f60_260.json
for m in HGA HGB; do (
  B=$V/recon_E14_best_B2A_S1_$m
  $PY $AR $O/${m}X_w1 33 $ANN 150 214 --base $B --face-thr 0.8 --iou 160,202,1 > $O/${m}X_w1.txt 2>&1
  $PY $AR $O/${m}X 33 $ANN 218 280 --base $O/${m}X_w1/bundle --face-thr 0.8 --iou 226,266,1 > $O/${m}X_w2.txt 2>&1
  mkdir -p $O/recon_E14_best_B2A_S1_${m}Xs15
  $PY $W/code/ep33_smooth_corr.py $B/$E $O/${m}X/bundle/$E $O/recon_E14_best_B2A_S1_${m}Xs15/$E --sigma 1.5 > $O/smooth_$m.txt 2>&1
  $PY $AR $O/rep_$m 33 $ANN 150 280 --base $O/recon_E14_best_B2A_S1_${m}Xs15 --face-thr 0.8 --report-only > $O/rep_new_$m.txt 2>&1
  $PY $AR $O/rep0_$m 33 $ANN 150 280 --base $B --face-thr 0.8 --report-only > $O/rep_old_$m.txt 2>&1
  $PY $W/code/ep33_facing_sheet.py $O/recon_E14_best_B2A_S1_${m}Xs15 $O/sheet_${m}_f180_214.jpg 180 214 2 $ANN --label "$m refit" > /dev/null 2>&1
  $PY $W/code/ep33_facing_sheet.py $O/recon_E14_best_B2A_S1_${m}Xs15 $O/sheet_${m}_f240_274.jpg 240 274 2 $ANN --label "$m refit" > /dev/null 2>&1
) & done; wait
tail -3 $O/smooth_HGA.txt
for m in HGA HGB; do echo "== $m marks ok (old -> new)"; for v in old new; do grep -cE "True\)?$|, True" $O/rep_${v}_$m.txt; done; done
