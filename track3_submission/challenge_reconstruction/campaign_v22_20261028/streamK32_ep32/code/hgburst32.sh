#!/bin/bash
# K32-HG burst fix chain (port of streamK33 hgburst.sh): wbeamHG with per-frame deg hinge, narrow windows, right hand only, node200.
# usage: hgburst32.sh V NAME TABLE [INIT|none] WINDOWS [DEP] [SEED]   (env: NC, LOOK, SQS, DH, TOL, OW, HG)
set -euo pipefail
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK32_ep32; V=$1; N=$2; T=$3; I=${4:-none}; W=$5; D=${6:-}; SEED=${7:-3211}; O=$S/runs/hg/$N; mkdir -p $O
IA=""; [ "$I" != none ] && IA="--init $I"; DA=""; [ -n "$D" ] && DA="--dependency=afterok:$D"
J=$(E31_TGT=$S/targets/$V sbatch --parsable $DA --nodelist=node200 --time=01:30:00 -c 16 -J K32HG-$N $S/code/job.sbatch wbeamHG.py --ep 32 --actor /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH12_ep32/runs/save_e1f0/actor --meshes /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep32_H12e1f0/episode_000032/meshes/episode_000032 --out $O --table $T $IA \
 --windows $W --beam 6 --n ${NC:-60} --per-parent 3 --look ${LOOK:-0.7} --pre 0.1 --post 0.05 --gap 1.6 --knots 2 --pos-std 0.004 --rot-std 0.08 --sq-std ${SQS:-0.15} \
 --hands wooden_spoon:right,blue_cup:none --rand-sides right --ilc 0.3,0.6,1.0,1.3 --ilc-sigmas 0.03,0.06 --tol ${TOL:-0.5,6} --ow "${OW:-{\"wooden_spoon\":3}}" --hinge ${HG:-8,1.0} --dhinge ${DH:-16,15} \
 --roll=wooden_spoon:right --roll-alphas 0.5,1.0,1.5,2.0 --sq 0.1,0.2,0.4,0.8 --adv 0.5,1,2,3 --adv-sides right --workers 16 --seed $SEED)
echo "$N job=$J windows=$W table=$T init=$I"; echo "$(date -u +%FT%TZ) HG $V $N job=$J windows=$W table=$T init=$I dh=${DH:-16,15}" >> $S/out/jobs_roll.txt
