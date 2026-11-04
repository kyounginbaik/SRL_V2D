#!/bin/bash
#SBATCH -p debug
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /data1/hansol/ep33_20261102/logs/orientfit2_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; W=/data1/hansol/ep33_20261102; O=$W/orient/fit2; V=/home/hansol/v2d/outputs/variants_20261030; E=episode_000033
AR=/home/hansol/v2d/outputs/ep32_robot_20261031/code/spoon_annot_roll.py; ANN=$W/orient/ep33_marks_merged_20261103.json
for m in HGA HGB; do (
  B=$V/recon_E14_best_B2A_S1_$m
  $PY $AR $O/${m}Y_w1 33 $ANN 150 230 --base $B --face-thr 0.8 --iou 160,226,1 > $O/${m}Y_w1.txt 2>&1
  $PY $AR $O/${m}Y 33 $ANN 236 290 --base $O/${m}Y_w1/bundle --face-thr 0.8 --iou 240,284,1 > $O/${m}Y_w2.txt 2>&1
  mkdir -p $O/recon_E14_best_B2A_S1_${m}Ys15
  $PY $W/code/ep33_smooth_corr.py $B/$E $O/${m}Y/bundle/$E $O/recon_E14_best_B2A_S1_${m}Ys15/$E --sigma 1.5 > $O/smooth_$m.txt 2>&1
  $PY $AR $O/rep_$m 33 $ANN 150 290 --base $O/recon_E14_best_B2A_S1_${m}Ys15 --face-thr 0.8 --report-only > $O/rep_new_$m.txt 2>&1
  $PY $AR $O/rep0_$m 33 $ANN 150 290 --base $B --face-thr 0.8 --report-only > $O/rep_old_$m.txt 2>&1
  $PY $W/code/ep33_facing_sheet.py $O/recon_E14_best_B2A_S1_${m}Ys15 $O/sheet_${m}_f180_224.jpg 180 224 2 $ANN --label "$m refit (user marks 11/03)" > /dev/null 2>&1
  $PY $W/code/ep33_facing_sheet.py $O/recon_E14_best_B2A_S1_${m}Ys15 $O/sheet_${m}_f240_280.jpg 240 280 2 $ANN --label "$m refit (user marks 11/03)" > /dev/null 2>&1
) & done; wait
for m in HGA HGB; do echo "== $m"; tail -1 $O/${m}Y_w1.txt | cut -c1-400; tail -1 $O/${m}Y_w2.txt | cut -c1-400
  for v in old new; do echo "$v ok: $(grep -c "'ok': True" $O/rep_${v}_$m.txt) / $(grep -c "'ok'" $O/rep_${v}_$m.txt)"; done; done
