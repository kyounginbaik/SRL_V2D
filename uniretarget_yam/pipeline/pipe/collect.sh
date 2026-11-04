#!/usr/bin/env bash
# Usage: ./collect.sh N   If the episode's best checkpoint holds at least 15/16 on both scoring runs for every object, make the three videos with the original inset and copy them to videos/successes/epNN.
N=$1; L=$(printf ep%02d $N); EP=$(printf recon_%06d $N); R=/scratch2/exaFLOPs/srl_yam_run; cd $R/pipe
RES=$(python3 - <<PY
import json
try:r=json.load(open('$R/comparison/results/upper_test_${RN:-$EP}_s42.json'))
except Exception:print('none');raise SystemExit
b={}
for x in r:b.setdefault(x['steps'],[]).append(x)
v={k:min(min(h['passed'] for h in y.get('holds',[y])) for y in u) for k,u in b.items()}
k=max(v,key=lambda s:(v[s],-sum(y['cm'] for y in b[s])));print(f"{v[k]:.3f} {k//10**6} {b[k][0]['cm']:.2f} {b[k][0]['deg']:.1f} {b[k][0]['object_max_rise_cm']:.1f}")
PY
)
set -- $RES; [ "$1" = none ] && { echo "$L not scored"; exit 0; }
PARTIAL=""; [ -f $R/search/$EP/partial.txt ] && PARTIAL=" PARTIAL_GATE"
python3 -c "import sys;sys.exit(0 if $1>=.9375 else 1)" || { echo "$L not a pass: best $RES$PARTIAL"; exit 0; }
D=$R/videos/successes/$L${VT:-}; mkdir -p $D; [ -n "$PARTIAL" ] && echo PARTIAL_GATE > $D/PARTIAL_GATE.txt
for n in Fixed_Reference_Low_Level UniRetarget_Physics_Result UniRetarget_Reference_Refinement; do ./pip.sh $N $n >/dev/null && cp $R/videos/$EP${VT:-}/${L}_${n}_With_Original.mp4 $D/; done
echo "$L PASS passed=$1 at $2M cm=$3 deg=$4 rise=$5$PARTIAL -> $D"
