#!/bin/bash
# K33-HG tail re-chain after the spin fix: 0.5 s windows, c3 settings + per-frame deg hinge. usage: hgtail.sh VAR NAME TABLE INIT WINDOWS DEP SEED
set -euo pipefail
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33; V=$1; N=$2; T=$3; I=$4; W=$5; D=${6:-}; SD=${7:-5501}; O=$S/runs/cw/$N; mkdir -p $O; DA=""; [ -n "$D" ] && DA="--dependency=afterok:$D"
E31_TGT=$S/targets/K33_cw${V}fix sbatch --parsable $DA -c 20 -J K33HG-${N//\//-} $S/code/job.sbatch wbeamHG.py --ep 33 --actor /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13a/runs/save_ep33_b1/actor --meshes /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep33_H13b1/episode_000033/meshes/episode_000033 --out $O --table $T --init $I \
 --windows $W --beam 4 --n 24 --per-parent 2 --look 1.2 --pre 0.15 --post 0.1 --gap 1.6 --knots 3 --pos-std 0.004 --rot-std 0.06 \
 --hands wooden_spoon:right,blue_cup:none --rand-sides right --ilc 0.3,0.6,1.0,1.3 --ilc-sigmas 0.03,0.06 --tol 0.5,6 --ow '{"wooden_spoon":3}' --hinge 8,1.0 --dhinge 8,20 \
 --roll=wooden_spoon:right --roll-alphas 0.5,1.0,1.5,2.0 --adv 0.5,0.9,1.3 --adv-sides right --workers 20 --seed $SD
