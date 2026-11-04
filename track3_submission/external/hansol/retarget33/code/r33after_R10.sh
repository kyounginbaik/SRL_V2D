#!/bin/bash
# wait for R10 g2, then eval + spoon-in-cup (vs c18) and the package chain
R=/data1/hansol/retarget33; cd $R; while squeue -h -j 69028 | grep -q .; do sleep 30; done
ls runs/cw/R10/g2/final_0.npz || exit 1
j=$(TG=R33_HGAc18 sbatch --parsable code/r33evcs2.sbatch R10 $R/runs/cw/R10/g2/final_0.npz); while squeue -h -j $j | grep -q .; do sleep 10; done; echo "EVAL_LOG logs/R33-evcs-$j.log"
bash code/r33pkg_R10.sh
