#!/usr/bin/env bash
# Usage: ./finish.sh N [prep_job]   After prep ends: train from the grasp that passed the Newton gate, else from the best final-stage grasp with lift >= .8 (flagged PARTIAL_GATE). Needs a grasp for every hand. prep_job clears that pilot's stuck train, score, baseline and video jobs.
N=$1; A=${2:-}; TS=42; EP=$(printf recon_%06d $N); R=/scratch2/exaFLOPs/srl_yam_run; S=$R/search/$EP; G=base_suma_rtx3090,dell_rtx3090,suma_rtx4090,gigabyte_a5000,asus_a5000,suma_a6000,gigabyte_a6000; P=base_suma_rtx3090,dell_rtx3090; cd $R/pipe
[ -z "$(squeue -u $USER -h -n p$N -o %i)" ] || { echo "ep$N: prep still running"; exit 1; }
L=$(ls -t p$N-*.log | head -1)
if grep -q "^PREP_DONE $N" $L; then echo "ep$N passed the gate: $(cat $S/pool.txt)"; rm -f $S/partial.txt; exit 0; fi
BEST=$(python3 - "$S" "$EP" /scratch2/exaFLOPs/SRL_V2D/uniretarget_yam/episodes/$EP/recon_meta.json <<'PY'
import sys,json,glob,os
S,EP,META=sys.argv[1:];b=None;n=len(json.load(open(META))['holds'])
fs=glob.glob(f'{S}/stage{n}_s*/check.txt') if n>1 else glob.glob(S+'/s*/check_*.txt')
for f in fs:
    r=[json.loads(l) for l in open(f) if l.startswith('{')]
    if not r or min(x.get('lift',0) for x in r)<.8:continue
    c=min(sum(v for k,v in x['kinds'].items() if k in('held','off_track'))/sum(x['kinds'].values()) for x in r);fr=min(x['frontier'] for x in r)
    d=os.path.dirname(f);name=os.path.basename(f);sub='flash_pass' if name in('check.txt','check_pass.txt') else 'flash_'+name[6:-4]
    if not os.path.isfile(f'{d}/{sub}/reference.parquet'):continue
    if b is None or (c,fr)>b[0]:b=((c,fr),f"/search/{EP}/{os.path.basename(d)}/{sub}/reference.parquet")
print(b[1] if b else '')
PY
)
if [ -n "$A" ]; then for i in 1 2 4 5; do J=$((A+i)); [ -n "$(squeue -h -j $J -o %j 2>/dev/null | grep -E "^(t$N|s$N|a$N)s$TS$|^v$N$")" ] && scancel $J; done; fi
[ -n "$BEST" ] || { echo "ep$N: not eligible, no grasp for every hand with lift >= .8"; exit 1; }
echo "$BEST" > $S/pool.txt; echo PARTIAL_GATE > $S/partial.txt; echo "ep$N PARTIAL_GATE: $BEST"
T=$(sbatch --parsable -J t${N}s$TS -p $G train.sbatch $N $TS); C=$(sbatch --parsable -J s${N}s$TS -p $P --dependency=afterok:$T score.sbatch $N $TS); B=$(sbatch --parsable -J a${N}s$TS -p $G train_a.sbatch $N $TS); V=$(sbatch --parsable -J v$N -p $P --dependency=afterany:$C,afterok:$B video.sbatch $N $TS); echo "train $T score $C baseline $B video $V"
