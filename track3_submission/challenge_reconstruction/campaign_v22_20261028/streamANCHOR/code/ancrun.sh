#!/bin/bash
# ANCHOR driver (copy of streamALL20K/code/a20run.sh): window-beam vs targets/$V (ANC_* f0-anchor-corrected targets) from the current donor table, then svst.
# svst (save actor with the best table -> unchanged stager -> G18 mesh check (== v29) -> build-v2 -> vs.sh scores).
# usage: [ACTOR=dir] ancrun.sh V EP NAME INIT_TABLE.npz|none [TOL] [HINGE] [WORKERS] [DEP]
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; S=$C22/streamALL20K
V=$1; EP=$2; N=$3; TB=${4:-none}; TOL=${5:-0.5,6}; HG=${6:-8,1.0}; WK=${7:-8}; DEP=${8:-}
A=${ACTOR:-$(python3 -c "import json;print(json.load(open('$S/out/v29_actors.json'))['$EP']['actors'][-1])")}   # ACTOR env: start actor override (ep35/ep33 fixed actors)
NM=$(printf episode_%06d $EP)
M=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/$(python3 -c "import json;print(json.load(open('$C22/selection_v29.json'))['$EP'])")/$NM/meshes/$NM
case $EP in
  1) H=white_pot:left,white_pot_lid:right; W0=1.5;; 3|5|6) H=wooden_spoon:right,white_pot:left; W0=1.5;;
  8) H=beige_cup:right,white_pot:left; W0=1.0;; 10) H=blue_cup:left,beige_cup:right; W0=1.0;;
  14) H=blue_cup:left,plastic_dish_rack:none; W0=0.5;; 15|16) H=wooden_spoon:right,plastic_dish_rack:none; W0=1.0;;
  19) H=mini_sweeper:right,mini_dust_pan:left; W0=0.8;; 20) H=blue_cup:left,water_pitcher:right; W0=0.5;;
  25) H=wooden_piece_1:left,wooden_piece_2:right; W0=1.0;; 28) H=white_pot:left,plastic_dish_rack:none; W0=0.5;;
  30) H=mini_sweeper:right,plastic_dish_rack:left; W0=1.0;; 32) H=wooden_spoon:right,blue_cup:left; W0=1.0;;
  33) H=wooden_spoon:right,blue_cup:left; W0=1.0;; 35|36|37) H=wooden_spoon:right; W0=1.0;; 38) H=mini_dust_pan:right; W0=1.0;;
esac
ROLL=""; case $H in *wooden_spoon:right*) ROLL="--roll wooden_spoon:right --roll-alphas 0.5,1.0,1.5,2.0 --sq 0.1,0.2";; esac
WIN=$(python3 -c "
import numpy as np
d=float(np.load('$A/training/policy.npz',allow_pickle=False)['duration_s']); w=[]; t=$W0
while t+0.5<d-0.05: w.append(f'{t:.2f}:{t+0.5:.2f}'); t+=0.5
w.append(f'{t:.2f}:{d-0.05:.2f}'); print(','.join(w))")
OUT=$S/runs/ep$EP/$N; mkdir -p $S/runs/ep$EP
TA=""; [ "$TB" != none ] && TA="--table $TB"
D=""; if [ -n "$DEP" ]; then case $DEP in *:*) D="--dependency=$DEP";; *) D="--dependency=afterany:$DEP";; esac; fi
J1=$(E31_TGT=$S/targets/$V sbatch --parsable $D -c $WK -J ANC-b-ep$EP-$N $S/code/job.sbatch wbeamH.py --ep $EP --actor $A --meshes $M --out $OUT $TA \
  --windows $WIN --beam 4 --n 16 --per-parent 2 --look 1.2 --pre 0.15 --post 0.1 --knots 3 --pos-std 0.004 --rot-std 0.06 --sq-std 0.04 \
  --scales 0.1,0.25,0.5,1.0 --rand-sides right,left --hands $H --ilc 0.3,0.6,1.0,1.3 --ilc-sigmas 0.03,0.06 --ilc-modes full,pos,rot,rotanc \
  --tol $TOL --mppe-w 1.0 --workers $WK --seed $((1000*EP+${#N})) --hinge $HG $ROLL --adv 0.5,0.9,1.3 --adv-g 0.5,1.0 --adv-sides right,both)
J2=$(sbatch --parsable --dependency=afterok:$J1 -J ANC-sv-ep$EP-$N $S/code/svst.sbatch $V $EP $N $A mresH.py $OUT/final_0.npz)
echo "ep$EP $N beam=$J1 svst=$J2 actor=$A out=$OUT"
echo "$(date -u +%FT%TZ) ep$EP $N beam=$J1 svst=$J2 table=$TB tol=$TOL hinge=$HG" >> $C22/streamANCHOR/out/jobs.txt
