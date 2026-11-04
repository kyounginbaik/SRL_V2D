#!/usr/bin/env bash
# Waits until the ep5 VOC runs have started and logged, or until the U runs have their 20M checks, then prints the assist level and upper checks.
cd /scratch2/exaFLOPs/srl_yam_run/pipe; O=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam/out/recon_000005; IDS=$(cat voc5.ids | tr ' ' ',')
for i in $(seq 1 480); do sleep 60; n=0; for g in U-g1-voc-seed42 U-g2-voc-seed42; do grep -q '"step": 200' $O/$g/upper_updates.jsonl 2>/dev/null && n=$((n+1)); done; [ $n = 2 ] && break; s=$(squeue -h -j $IDS -o %i 2>/dev/null) || continue; [ -z "$s" ] && break; done
for t in t5g1v t5g2v a5v; do L=$(ls -t $t-*.log 2>/dev/null | head -1); [ -n "$L" ] && { echo "== $L"; grep -E "Traceback|^[A-Za-z]*Error|TRAIN" $L | tail -2 | cut -c1-200; }; done
python3 - <<PY
import json,os
for g in ('U-g1-voc-seed42','U-g2-voc-seed42','A-voc-seed42'):
    d='$O/'+g;m=d+'/training_metrics.jsonl';f=d+'/upper_updates.jsonl';print(g)
    if os.path.exists(m):
        last=None;seen=[]
        for l in open(m):
            try:r=json.loads(l)
            except Exception:continue
            v=r.get('curriculum/voc_scale')
            if v is not None and (not seen or seen[-1]!=round(v,3)):seen.append(round(v,3))
            last=r
        print('  voc scales seen',seen,'last step',(last or {}).get('step'))
    if os.path.exists(f):
        for x in map(json.loads,open(f)):
            b=x['baseline'];print('  ',x['step']//10**6,'M acc',x['accepted'],'pass',round(b['passed'],2),'fr',b['frontier'],b['kinds'],'cm',b['cm'],[(round(h['passed'],2),h['cm']) for h in b.get('holds',[])])
PY
date
