#!/bin/bash
# usage: r33after_eval.sh NAME LAST_JOB  -> waits, then eval vs c18 + spoon-in-cup + contact/slip diagnostic
R=/data1/hansol/retarget33; cd $R; N=$1; while squeue -h -j $2 | grep -q .; do sleep 30; done; ls runs/cw/${N}_D/final_0.npz || exit 1
j=$(TG=R33_HGAc18 FZ=$FZ sbatch --parsable code/r33evcs2.sbatch $N $R/runs/cw/${N}_D/final_0.npz); while squeue -h -j $j | grep -q .; do sleep 10; done; echo "EVAL_LOG logs/R33-evcs-$j.log"
P=/home/hansol/v2d/.venvs/v2d-host/bin/python; PP=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
j=$(sbatch --parsable -p debug --exclude=node200 -c 1 --mem=6G --time=00:15:00 -J R33-contact -o $R/logs/%x-%j.log --wrap "export PYTHONPATH=$PP; cd $R/tmp; $P -B $R/code/r33contact.py eval_$N"); while squeue -h -j $j | grep -q .; do sleep 10; done; echo "CONTACT_LOG logs/R33-contact-$j.log"
