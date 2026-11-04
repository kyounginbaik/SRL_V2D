#!/bin/bash
# usage: prog.sh DIR... : print initial and latest best of kcem runs
for d in "$@"; do python3 - "$d" <<'P'
import json,sys,os
d=sys.argv[1]
try: i=json.load(open(d+'/initial.json')); s=f"init {[round(x,2) for x in i['mean_position_cm']]} {[round(x,1) for x in i['mean_orientation_deg']]}"
except Exception: s='init ?'
try:
    h=json.load(open(d+'/progress.json'))['history'][-1]; b=h['best']; s+=f" | it{h['iteration']} {[round(x,2) for x in b['mean_position_cm']]} {[round(x,1) for x in b['mean_orientation_deg']]} cost {b['cost']:.3f} t{h['elapsed_s']:.0f}"
except Exception: pass
if os.path.exists(d+'/result.json'): s+=' DONE'
print(os.path.basename(d), s)
P
done
