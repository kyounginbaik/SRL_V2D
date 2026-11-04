#!/bin/bash
# ep33 spoon tilt fix in two windows (f156-190, f224-264) on HGA / HGB, full and half strength; then checks.
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; M=/data1/hansol/mink_S1_20261031; C=$M/code; W=/data1/hansol/ep33_20261102; R=$W/tiltfix
V=/home/hansol/v2d/outputs/variants_20261030; S=$V/recon_E14_best_B2A_S1
e=33; E=episode_000033; CT=blue_cup; ANN=/home/hansol/v2d/outputs/ep32_robot_20261031/annot_json/ep33_f60_260.json
rm -rf $R; mkdir -p $R
for b in HGA HGB; do for sc in 10 05; do m=${b}T$sc
  $PY $W/code/ep33_tiltfix.py $V/recon_E14_best_B2A_S1_$b $e $W/depthline/ep33.json $R/fit_$m/$E --windows 156-190,224-264 --scale ${sc:0:1}.${sc:1:1} > $R/fit_$m.txt 2>&1
  X=$R/recon_E14_best_B2A_S1_$m; mkdir -p $X; cp -a $V/recon_E14_best_B2A_S1_$b/$E $X/; cp $R/fit_$m/$E/object_poses.npz $R/fit_$m/$E/tiltfix_report.json $X/$E/
done; done
L=""; for m in HGA HGB HGAT10 HGAT05 HGBT10 HGBT05; do case $m in HGA|HGB) X=$V/recon_E14_best_B2A_S1_$m;; *) X=$R/recon_E14_best_B2A_S1_$m;; esac; L="$L,$m=$X"
  $PY /home/hansol/v2d/outputs/ep32_robot_20261031/code/smooth_check.py $e $ANN $S $X/$E $R/targets_$m > $R/check_$m.txt 2>&1 &
  $PY $C/contact_jitter.py $X $e $CT > $R/cj_$m.txt 2>&1 &
  $PY $C/handle_line.py $X $e $CT $R/line_$m > $R/line_$m.txt 2>&1 &
done; wait
$PY $C/handle_depth_line.py ${L:1} $e $CT $R/dl_all > $R/dl_all.txt 2>&1
echo TILTFIX-DONE
