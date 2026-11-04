#!/bin/bash
# K33-HG burst fix chain: wbeamHG with per-frame deg hinge, narrow windows, right hand only. usage: hgburst.sh VAR NAME TABLE [INIT] WINDOWS [DEP] [SEED]
set -euo pipefail
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33; V=$1; N=$2; T=$3; I=$4; W=$5; D=${6:-}; SEED=${7:-3311}; O=$S/runs/cw/$N; mkdir -p $O
IA=""; [ "$I" != none ] && IA="--init $I"; DA=""; [ -n "$D" ] && DA="--dependency=afterok:$D"
E31_TGT=$S/targets/K33_cw${V}fix sbatch --parsable $DA -c 20 -J K33HG-${N//\//-} $S/code/job.sbatch wbeamHG.py --ep 33 --actor /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13a/runs/save_ep33_b1/actor --meshes /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep33_H13b1/episode_000033/meshes/episode_000033 --out $O --table $T $IA \
 --windows $W --beam 6 --n ${NC:-60} --per-parent 3 --look ${LOOK:-0.7} --pre 0.1 --post 0.05 --gap 1.6 --knots 2 --pos-std 0.004 --rot-std 0.08 --sq-std ${SQS:-0.15} \
 --hands wooden_spoon:right,blue_cup:none --rand-sides right --ilc 0.3,0.6,1.0,1.3 --ilc-sigmas 0.03,0.06 --tol 0.5,6 --ow '{"wooden_spoon":3}' --hinge 8,1.0 --dhinge ${DH:-16,15} \
 --roll=wooden_spoon:right --roll-alphas 0.5,1.0,1.5,2.0 --sq 0.1,0.2,0.4,0.8 --adv 0.5,1,2,3 --adv-sides right --workers 20 --seed $SEED
