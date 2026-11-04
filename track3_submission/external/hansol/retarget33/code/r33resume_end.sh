#!/bin/bash
# wait for R5_end (68852); if it timed out before its FINAL, continue from its last saved beam over the remaining windows
R=/data1/hansol/retarget33; cd $R
while squeue -h -j 68852 | grep -q .; do sleep 30; done
f=$(ls $R/logs/*-68852.log); if grep -q "^FINAL" $f; then echo "R5_end finished with FINAL, no resume"; exit 0; fi
last=$(ls $R/runs/cw/R5_end/beam_w*.npz | sort | tail -1); k=$(basename $last .npz | sed 's/beam_w//'); k=$((10#$k))
ALL="14.00:14.25,14.25:14.50,14.50:14.75,14.75:15.00,15.00:15.25,15.25:15.50,15.50:15.75,15.75:16.00,16.00:16.25,16.25:16.50,16.50:16.75,16.75:17.00,17.00:17.25,17.25:17.30"
REST=$(echo $ALL | tr ',' '\n' | tail -n +$((k + 2)) | paste -sd,); echo "resume from $last (w$k), windows $REST"
export TG=R33_HGA_r6 WB=wbeamHGs.py SH=2,3,4,55,300 HANDS=wooden_spoon:right,blue_cup:left RS=right,left ADVS=right,both HG=16,0.8
J=$(bash code/r33hg.sh burst R5_end2 $R/runs/cw/R4A/g2/final_0.npz $last $REST "" 3354); echo "R5_end2 job $J"
