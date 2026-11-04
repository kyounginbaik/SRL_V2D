#!/bin/bash
# T33 copy of retarget33 r33hg.sh: ACT= actor override (default H13), TABLE none allowed, target default R33_HGAc18, outputs under teacher33. R33 second pass (copy of the team's hgburst.sh / hgtail.sh settings, target = our R33_HGA7b). usage: r33hg.sh burst|tail NAME TABLE INIT|none WINDOWS [DEP] [SEED]
set -euo pipefail
R=/data1/hansol/teacher33; MODE=$1; N=$2; T=$3; I=$4; W=$5; D=${6:-}; SEED=${7:-3311}; O=$R/runs/cw/$N; mkdir -p $O
A=${ACT:-/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13a/runs/save_ep33_b1/actor}
M=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep33_H13b1/episode_000033/meshes/episode_000033
IA=""; [ "$I" != none ] && IA="--init $I"; DA=""; [ -n "$D" ] && DA="--dependency=afterok:$D"
OWD='{"wooden_spoon":3}'; OWV=${OW:-$OWD}   # T33: OW= object weights override (default spoon 3)
COMMON="--ep 33 --actor $A --meshes $M --out $O $( [ $T = none ] || echo --table $T ) $IA --windows $W --hands ${HANDS:-wooden_spoon:right,blue_cup:none} --rand-sides ${RS:-right} --ilc 0.3,0.6,1.0,1.3 --ilc-sigmas 0.03,0.06 --tol 0.5,6 --ow $OWV --hinge ${HG:-8,1.0} --roll=wooden_spoon:right --roll-alphas 0.5,1.0,1.5,2.0 --adv-sides ${ADVS:-right} --workers ${WK:-20} --seed $SEED"
if [ $MODE = burst ]; then X="--beam 6 --n ${NC:-40} --per-parent 3 --look ${LOOK:-0.7} --pre 0.1 --post 0.05 --gap 1.6 --knots 2 --pos-std ${PSD:-0.004} --rot-std ${RSD:-0.08} --sq-std ${SQS:-0.15} --dhinge ${DH:-16,15} --sq 0.1,0.2,0.4,0.8 --adv 0.5,1,2,3"
else X="--beam 4 --n 24 --per-parent 2 --look 1.2 --pre 0.15 --post 0.1 --gap 1.6 --knots 3 --pos-std 0.004 --rot-std 0.06 --dhinge 8,20 --adv 0.5,0.9,1.3"; fi
J=$(E31_TGT=/data1/hansol/retarget33/targets/${TG:-R33_HGAc18} sbatch --parsable $DA -c ${WK:-20} -J T33HG-$N $R/code/job.sbatch ${WB:-wbeamHG.py} $COMMON $X ${SH:+--shinge $SH} ${RH:+--rhinge $RH} ${GH:+--ghinge $GH} ${FZ:+--freeze $FZ})
echo "$(date -u +%FT%TZ) ${TG:-R33_HGAc18} $N $MODE act=$A job=$J table=$T init=$I windows=$W" >> $R/cw/jobs.txt; echo $J
