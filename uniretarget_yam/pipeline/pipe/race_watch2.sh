#!/usr/bin/env bash
# Waits until grips g0 and g2 have their 20M check or have stopped, then prints their updates.
cd /scratch2/exaFLOPs/srl_yam_run/pipe; O=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam/out/recon_000005
for i in $(seq 1 300); do sleep 60; n=0; for g in U-g0-seed42 U-g2-seed42; do grep -q '"step": 200' $O/$g/upper_updates.jsonl 2>/dev/null && n=$((n+1)); done; [ $n = 2 ] && break; s=$(squeue -u $USER -h -o %j) || continue; [ -z "$(echo "$s" | grep -E '^t5g[02]$')" ] && break; done
python3 - <<PY
import json,os
for g in ('g0','g2'):
    f='$O/U-'+g+'-seed42/upper_updates.jsonl'
    if not os.path.exists(f):print(g,'no updates');continue
    for x in map(json.loads,open(f)):
        b=x['baseline'];print(g,x['step']//10**6,'M acc',x['accepted'],'pass',round(b['passed'],2),'fr',b['frontier'],b['kinds'],'cm',round(b['cm'],2),'holds',[(round(h['passed'],2),round(h['cm'],1)) for h in b.get('holds',[])])
PY
date
