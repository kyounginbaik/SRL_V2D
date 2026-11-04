#!/bin/bash
R=/data1/hansol/retarget33; cd $R; while squeue -h -j 69118 | grep -q .; do sleep 30; done; ls runs/cw/R12_D/final_0.npz || exit 1
j=$(TG=R33_HGAc18 sbatch --parsable code/r33evcs2.sbatch R12 $R/runs/cw/R12_D/final_0.npz); while squeue -h -j $j | grep -q .; do sleep 10; done; echo "EVAL_LOG logs/R33-evcs-$j.log"
bash code/r33pkg_R12.sh
