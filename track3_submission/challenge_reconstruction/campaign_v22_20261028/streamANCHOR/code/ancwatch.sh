#!/bin/bash
# ANCHOR watcher: for each "V EP NAME" pair, wait until its build-v2 candidate is complete (svst done), then run ancpost.sh once.
C=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; A=$C/streamANCHOR
PAIRS="ANC_ep16_rack:16:anc1 ANC_ep15_rack:15:anc1 ANC_ep28_rack:28:anc1 ANC_ep38_pan:38:anc1 ANC_ep37_spoon:37:anc1 ANC_ep5_pot:5:anc1 ANC_ep14_rack:14:anc1 ANC_ep20_pitcher:20:anc1 ANC_ep8_potall:8:anc2 ANC_ep38_panall:38:anc2 ANC_ep5_potall:5:anc2 ANC_ep20_pitchall:20:anc2"
declare -A DONE
while true; do
  left=0
  for p in $PAIRS; do IFS=: read V E N <<< "$p"; K=$E-$N; [ -n "${DONE[$K]:-}" ] && continue; left=1
    TAG=A20_ep${E}_$N; CJ=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/$TAG/$(printf episode_%06d $E)/completed.json
    J=$(grep -h "ep$E $N " $A/out/jobs.txt | tail -1 | grep -o 'svst=[0-9]*' | cut -d= -f2)
    st=$(squeue -h -j $J -o %T 2>/dev/null)
    if [ -f $CJ ] && [ -z "$st" ]; then DONE[$K]=1; echo "=== ep$E $N $(date -u +%H:%M)"; bash $A/code/ancpost.sh $V $E $N 2>&1 | tail -8; 
    elif [ -z "$st" ] && [ ! -f $CJ ] && squeue -u exaFLOPs -h >/dev/null 2>&1 && ! squeue -u exaFLOPs -h -o %j | grep -q "ep$E-$N"; then DONE[$K]=1; echo "=== ep$E $N FAILED (no candidate) $(date -u +%H:%M)"; fi
  done
  [ $left = 0 ] && break; sleep 60
done
echo ALLDONE
