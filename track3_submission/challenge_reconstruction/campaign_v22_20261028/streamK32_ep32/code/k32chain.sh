#!/bin/bash
# K32: chained window-beam (wbeamH, phase residuals on ACTOR) vs targets/$V, split into NG jobs of <= 8 CPUs (ALL20 holds 40 CPUs):
# job g runs window group g with --init = previous group's last beam_wNN.npz (beam state carried; same algorithm as one long beam).
# usage: k32chain.sh V NAME ACTOR [INIT_BEAM.npz|none] [TOL] [HINGE] [W0] [WORKERS] [NG] [OW]
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; S=$C22/streamK32_ep32
V=$1; N=$2; A=$3; IB=${4:-none}; TOL=${5:-0.6,7}; HG=${6:-6,1.0}; W0=${7:-1.0}; WK=${8:-8}; NG=${9:-4}; OW=${10:-'{}'}
M=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep32_H12e1f0/episode_000032/meshes/episode_000032
GROUPS_=$(python3 -c "
import numpy as np
d=float(np.load('$A/training/policy.npz',allow_pickle=False)['duration_s']); w=[]; t=$W0
while t+0.5<d-0.05: w.append(f'{t:.2f}:{t+0.5:.2f}'); t+=0.5
w.append(f'{t:.2f}:{d-0.05:.2f}'); k=int(np.ceil(len(w)/$NG)); print(' '.join(','.join(w[i:i+k]) for i in range(0,len(w),k)))")
PREV=""; INIT=$IB; g=0; OUT=$S/runs/$N
for WIN in $GROUPS_; do
  g=$((g+1)); O=$OUT/g$g; mkdir -p $O
  NW=$(echo $WIN | tr ',' '\n' | wc -l)
  IA=""; [ "$INIT" != none ] && IA="--init $INIT"
  D=""; [ -n "$PREV" ] && D="--dependency=afterok:$PREV"
  J=$(E31_TGT=$S/targets/$V sbatch --parsable $D -c $WK -J K32-$N-g$g $S/code/job.sbatch wbeamH.py --ep 32 --actor $A --meshes $M --out $O $IA \
    --windows $WIN --beam 4 --n 16 --per-parent 2 --look 1.2 --pre 0.15 --post 0.1 --knots 3 --pos-std 0.004 --rot-std 0.06 --sq-std 0.04 \
    --scales 0.1,0.25,0.5,1.0 --rand-sides right,left --hands wooden_spoon:right,blue_cup:left --ilc 0.3,0.6,1.0,1.3 --ilc-sigmas 0.03,0.06 --ilc-modes full,pos,rot,rotanc \
    --tol $TOL --ow "$OW" --mppe-w 1.0 --workers $WK --seed $((32000+g+${#N})) --hinge $HG --roll wooden_spoon:right --roll-alphas ${RA:-0.5,1.0,1.5,2.0} --sq 0.1,0.2 \
    --adv 0.5,0.9,1.3 --adv-g 0.5,1.0 --adv-sides right,both)
  echo "g$g job=$J windows=$WIN"; echo "$(date -u +%FT%TZ) $N g$g job=$J init=$INIT windows=$WIN tol=$TOL hinge=$HG ow=$OW" >> $S/out/jobs.txt
  PREV=$J; INIT=$O/beam_w$(printf %02d $((NW-1))).npz
done
echo "LAST=$PREV OUT=$OUT/g$g"
