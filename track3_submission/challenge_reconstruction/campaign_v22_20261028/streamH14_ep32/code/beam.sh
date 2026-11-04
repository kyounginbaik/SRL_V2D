#!/bin/bash
# usage: beam.sh NAME CPUS WINFILE INIT(none|npz) WIN_FROM extra args...   -> sbatch wbeamH14 on node01 (out runs/NAME)
H=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH14_ep32; C=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028
N=$1; CP=$2; WF=$3; IN=$4; WFROM=$5; shift 5
A=${ACTOR:-$C/streamH12_ep32/runs/save_e1f0/actor}; M=$C/streamH12_ep32/staging/ep32_e1f0/components/episode_000032/meshes/episode_000032
IA=""; [ "$IN" != none ] && IA="--init $IN"
sbatch --cpus-per-task $CP --mem 64G --job-name H14-$N $H/code/job.sbatch wbeamH14.py --actor $A --meshes $M --out $H/runs/$N --windows $(cat $H/code/$WF) --win-from $WFROM \
  --beam 6 --n 8 --look 1.0 --pre 0.15 --post 0.1 --gap 1.6 --knots 3 --pos-std 0.004 --rot-std 0.08 --sq-std 0.04 --ilc 0.3,0.6,1.0,1.3 --ilc-sigmas 0.06,0.15 \
  --cup-w 0.5 --lilc 0.3,0.6,1.0 --workers $CP --per-parent 2 --adv 1,2,4,6 --adv-g 0.5,1.0 --sq 0.1,0.2,0.4 --hinge 4,2.0 $IA "$@"
