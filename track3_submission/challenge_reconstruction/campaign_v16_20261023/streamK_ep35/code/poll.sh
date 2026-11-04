#!/bin/bash
# summary of ILC logs and distilled grids. usage: poll.sh [ilc-glob-prefixes...]
K=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35
squeue -u exaFLOPs -h -o "%j %T" | grep K35 | sort | uniq -c
for f in $K/logs/${1:-ilc}*.log; do n=$(grep -c '^{' $f); [ $n -gt 0 ] && echo "$(basename $f .log) $(grep -h '"out"\|"iteration"' $f | python3 -c "
import sys,json,os
for l in sys.stdin:
    r=json.loads(l)
    if 'iteration' in r: print(f\"it{r['iteration']} {r['cm']}/{r['deg']} sr{int(r['mp_sr'])}\", end='  ')
    else: print(os.path.basename(r['out']), r['objects'].get('wooden_spoon'), end='  ')
")"; done
