#!/usr/bin/env bash
# Usage: ./race_watch.sh prep_job   Waits for the race prep to end, prints it, then waits until every grip has its 20M check or has stopped, and prints the comparison.
cd /scratch2/exaFLOPs/srl_yam_run/pipe; RP=$1; O=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam/out/recon_000005
for i in $(seq 1 600); do [ -z "$(squeue -h -j $RP -o %i 2>/dev/null)" ] && break; sleep 60; done
grep -E "NEWTON|NO_REFERENCE|RACE_PREP_DONE|Traceback" rp5-$RP.log | cut -c1-400
for i in $(seq 1 900); do sleep 60; n=0; for g in U-seed42 U-g0-seed42 U-g2-seed42; do grep -q '"step": 20' $O/$g/upper_updates.jsonl 2>/dev/null && n=$((n+1)); done; r=$(squeue -u $USER -h -o %j | grep -cE '^(t5s42|t5g0|t5g2)$'); { [ $n = 3 ] || [ $r = 0 ]; } && break; done
python3 - <<PY
import json,glob,os
for g,l in (('g0','U-g0-seed42'),('g1','U-seed42'),('g2','U-g2-seed42')):
    f='$O/'+l+'/upper_updates.jsonl'
    if not os.path.exists(f):print(g,'no updates');continue
    for x in map(json.loads,open(f)):
        b=x['baseline'];print(g,x['step']//10**6,'M accepted',x['accepted'],'v',x.get('version_after'),'passed',round(b['passed'],2),'frontier',b['frontier'],'kinds',b['kinds'],'cm',round(b['cm'],2),'lift',round(b.get('lift',0),2))
PY
date
