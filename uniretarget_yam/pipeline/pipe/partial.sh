#!/usr/bin/env bash
# Usage: ./partial.sh N [min_held_fraction]  When every seed of the last stage missed the Newton gate, train on the best grasp found: the policy can firm up a grasp that holds some of the time.
N=$1; MIN=${2:-0.25}; TS=42; EP=$(printf recon_%06d $N); S=/scratch2/exaFLOPs/srl_yam_run/search/$EP; G=base_suma_rtx3090,dell_rtx3090,suma_rtx4090,gigabyte_a5000,asus_a5000,suma_a6000,gigabyte_a6000; P=base_suma_rtx3090,dell_rtx3090; cd /scratch2/exaFLOPs/srl_yam_run/pipe
BEST=$(python3 - "$S" "$MIN" <<'PY'
import sys,json,glob,os
S,m=sys.argv[1],float(sys.argv[2]);b=None
for f in glob.glob(S+'/stage*_s*/check.txt'):
    r=[json.loads(l) for l in open(f) if l.startswith('{')]
    if not r:continue
    v=min(sum(n for k,n in x['kinds'].items() if k in('held','off_track'))/sum(x['kinds'].values()) for x in r)
    if v>=m and min(x.get('lift',0) for x in r)>=.8 and (b is None or v>b[0]):b=(v,os.path.dirname(f))
print(b[1] if b else '')
PY
)
[ -n "$BEST" ] || { echo "no stage reached $MIN carried"; exit 1; }
echo "/search/$EP/$(basename $BEST)/flash_pass/reference.parquet" > $S/pool.txt; echo PARTIAL_GATE > $S/partial.txt; echo "training ep$N on $BEST"
T=$(sbatch --parsable -J t${N}s$TS -p $G train.sbatch $N $TS); C=$(sbatch --parsable -J s${N}s$TS -p $P --dependency=afterok:$T score.sbatch $N $TS); A=$(sbatch --parsable -J a${N}s$TS -p $G train_a.sbatch $N $TS); V=$(sbatch --parsable -J v$N -p $P --dependency=afterok:$C:$A video.sbatch $N $TS); echo "train $T score $C baseline $A video $V"
