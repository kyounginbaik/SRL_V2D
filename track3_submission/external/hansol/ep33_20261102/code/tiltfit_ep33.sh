#!/bin/bash
# ep33 spoon: gated stereo-depth tilt on top of HGA / HGB (only near frames where the offset-corrected depth tilt differs
# from the bundle by > 6 deg; running median +-2). sigma 4 and 2. Then image / annotation / depth / jitter checks.
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages OMP_NUM_THREADS=1
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; M=/data1/hansol/mink_S1_20261031; C=$M/code; W=/data1/hansol/ep33_20261102; R=$W/tilt
V=/home/hansol/v2d/outputs/variants_20261030; S=$V/recon_E14_best_B2A_S1
e=33; E=episode_000033; CT=blue_cup; ANN=/home/hansol/v2d/outputs/ep32_robot_20261031/annot_json/ep33_f60_260.json
mkdir -p $R
for b in HGA HGB; do $PY $C/handle_line.py $V/recon_E14_best_B2A_S1_$b $e $CT $R/line_$b > $R/line_$b.txt 2>&1 & done; wait
for b in HGA HGB; do for sg in 4 2; do m=${b}D$sg
  $PY $C/handle_smooth_fit.py $V/recon_E14_best_B2A_S1_$b $e $R/line_$b.json $R/fit_$m/$E --sig-acc 2 --linear --gate 4 \
      --depth $W/depthline/ep33.json --sig-tilt $sg --depth-gate 6 --depth-smooth 2 > $R/fit_$m.txt 2>&1 &
done; done; wait
for b in HGA HGB; do for sg in 4 2; do m=${b}D$sg; X=$R/recon_E14_best_B2A_S1_$m; rm -rf $X; mkdir -p $X
  cp -a $V/recon_E14_best_B2A_S1_$b/$E $X/; cp $R/fit_$m/$E/object_poses.npz $R/fit_$m/$E/handle_smooth_fit_report.json $X/$E/
done; done
for m in HGA HGB HGAD4 HGAD2 HGBD4 HGBD2; do case $m in HGA|HGB) X=$V/recon_E14_best_B2A_S1_$m;; *) X=$R/recon_E14_best_B2A_S1_$m;; esac
  $PY /home/hansol/v2d/outputs/ep32_robot_20261031/code/smooth_check.py $e $ANN $S $X/$E $R/targets_$m > $R/check_$m.txt 2>&1 &
  $PY $C/contact_jitter.py $X $e $CT > $R/cj_$m.txt 2>&1 &
done; wait
$PY $C/handle_depth_line.py HGA=$V/recon_E14_best_B2A_S1_HGA,HGAD4=$R/recon_E14_best_B2A_S1_HGAD4,HGAD2=$R/recon_E14_best_B2A_S1_HGAD2,HGB=$V/recon_E14_best_B2A_S1_HGB,HGBD4=$R/recon_E14_best_B2A_S1_HGBD4,HGBD2=$R/recon_E14_best_B2A_S1_HGBD2 $e $CT $R/dl_all > $R/dl_all.txt 2>&1
echo TILT-DONE
