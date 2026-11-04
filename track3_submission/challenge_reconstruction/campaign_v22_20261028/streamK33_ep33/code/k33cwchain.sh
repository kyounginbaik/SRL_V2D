#!/bin/bash
# K33-cw: chained window-beam (wbeamH, phase residuals on ACTOR [+TABLE]) vs targets/$V for ep33, split into NG jobs of WK CPUs
# (copy of streamK32_ep32/code/k32chain.sh; ep33 hands as K33 b5). Job g runs window group g with --init = previous group's last beam.
# usage: k33cwchain.sh V NAME ACTOR TABLE|none [INIT_BEAM|none] [TOL] [HINGE] [W0] [WORKERS] [NG] [OW] [DEP_JOB]
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; S=$C22/streamK33_ep33
V=$1; N=$2; A=$3; TB=$4; IB=${5:-none}; TOL=${6:-0.6,7}; HG=${7:-6,1.0}; W0=${8:-2.25}; WK=${9:-8}; NG=${10:-5}; OW=${11:-'{}'}; DEP0=${12:-}
M=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep33_H13b1/episode_000033/meshes/episode_000033
GROUPS_=$(python3 -c "
import numpy as np
d=float(np.load('$A/training/policy.npz',allow_pickle=False)['duration_s']); w=[]; t=$W0
while t+0.5<d-0.05: w.append(f'{t:.2f}:{t+0.5:.2f}'); t+=0.5
w.append(f'{t:.2f}:{d-0.05:.2f}'); k=int(np.ceil(len(w)/$NG)); print(' '.join(','.join(w[i:i+k]) for i in range(0,len(w),k)))")
PREV=$DEP0; INIT=$IB; g=0; OUT=$S/runs/cw/$N; TA=""; [ "$TB" != none ] && TA="--table $TB"
for WIN in $GROUPS_; do
  g=$((g+1)); O=$OUT/g$g; mkdir -p $O
  NW=$(echo $WIN | tr ',' '\n' | wc -l)
  IA=""; [ "$INIT" != none ] && IA="--init $INIT"
  D=""; [ -n "$PREV" ] && D="--dependency=afterok:$PREV"
  J=$(E31_TGT=$S/targets/$V sbatch --parsable $D -c $WK -J K33cw-$N-g$g $S/code/job.sbatch wbeamH.py --ep 33 --actor $A --meshes $M --out $O $TA $IA \
    --windows $WIN --beam 4 --n 16 --per-parent 2 --look 1.2 --pre 0.15 --post 0.1 --gap 1.6 --knots 3 --pos-std 0.004 --rot-std 0.06 \
    --hands ${HANDS:-wooden_spoon:right,blue_cup:left} --rand-sides ${RS:-right,left} --ilc 0.3,0.6,1.0,1.3 --ilc-sigmas 0.03,0.06 \
    --tol $TOL --ow "$OW" --hinge $HG --roll="${ROLL-wooden_spoon:right}" --roll-alphas 0.5,1.0,1.5,2.0 \
    --adv 0.5,0.9,1.3 --adv-sides ${ADVS:-right,both} --workers $WK --seed $((33100+g+${#N})))
  echo "g$g job=$J windows=$WIN"; mkdir -p $S/cw; echo "$(date -u +%FT%TZ) $V $N g$g job=$J table=$TB init=$INIT windows=$WIN tol=$TOL hinge=$HG ow=$OW" >> $S/cw/jobs.txt
  PREV=$J; INIT=$O/beam_w$(printf %02d $((NW-1))).npz
done
echo "LAST=$PREV OUT=$OUT/g$g"
