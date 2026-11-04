#!/bin/bash
R=/data1/hansol/retarget33; cd $R
j=$(TG=R33_HGAc18 sbatch --parsable code/r33evcs2.sbatch R11 $R/runs/cw/R11_C/final_0.npz); while squeue -h -j $j | grep -q .; do sleep 10; done; echo "EVAL_LOG logs/R33-evcs-$j.log"
bash code/r33pkg_R11.sh
