#!/bin/bash
# usage: rank.sh JOBID [N]
cd /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep19/logs
for f in F19-port-$1_*.log; do grep -o '"out": "[^"]*\|"vs_compose": {[^}]*}' $f | sed 's#.*/port/##;s/"vs_compose": //;s/"mini_//g' | tr '\n' ' '; echo; done | python3 -c "
import sys,re
rows=[]
for l in sys.stdin:
    m=re.findall(r'\[([\d.]+), ([\d.]+)\]',l)
    if len(m)<2: print('INCOMPLETE',l.strip()); continue
    (a,b),(c,d)=[(float(x),float(y)) for x,y in m]; rows.append((max(a/2.4,b/24,c/2.4,d/24),l.split()[0],a,b,c,d))
for r in sorted(rows)[:int(sys.argv[1])]: print('%.3f %-40s sw %.2f/%.1f dp %.2f/%.1f'%r)
" ${2:-15}
