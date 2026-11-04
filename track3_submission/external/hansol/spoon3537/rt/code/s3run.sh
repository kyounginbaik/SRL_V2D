#!/bin/bash
# spoon3537 launcher (A20 recipe, paths under /data1/hansol/spoon3537/rt): window-beam (wbeamH on the FINAL actor, FINAL beam table as
# --table, only WINDOWS re-searched) vs targets/TARGET_LABEL -> s3svst (mresH save -> stager -> G18 swap -> build-v2) -> post job:
# proxy scores (new candidate and FINAL candidate, vs new target and vs tgt_v32) + s3pkg splice into the combined final package.
# usage: [WK=cpus] [TL=hh:mm:ss] [INIT=<run>/beam_wNN.npz] [BASEP=<base parquet>] [PKGLABEL=<label>] s3run.sh EP NAME TARGET_LABEL WINDOWS [DEP]
#   WINDOWS = "a:b" range in s, split into 0.5 s beam windows. Must be ONE contiguous range per beam run (wbeamH raises
#   "prefix mismatch" across a gap); to add a later range, continue a finished run with INIT=<that run>/beam_wNN.npz.
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; R=/data1/hansol/spoon3537/rt; C=$R/code
EP=$1; N=$2; V=$3; RANGES=$4; DEP=${5:-}; NM=$(printf episode_%06d $EP)
case $EP in
  3) A=$C22/streamH13b/runs/ep3/save_a3b/actor; TB=$C22/streamALL20K/runs/ep3/b2/final_0.npz; FC=A20_ep3_b2; TOL=0.5,6; HG=8,1.0; HANDS=wooden_spoon:right,white_pot:left ;;
  36) A=$C22/streamH13a/runs/save_ep36_b1/actor; TB=$C22/streamALL20K/runs/ep36/b2/final_0.npz; FC=A20_ep36_b2; TOL=0.5,6; HG=8,1.0 ;;
  37) A=$C22/streamH13a/runs/save_ep37_b1/actor; TB=$C22/streamALL20K/runs/ep37/b1/final_0.npz; FC=A20_ep37_b1; TOL=0.6,7; HG=6,1.0 ;;
  *) echo "EP $EP not supported (ep35 = K35 stream with its own stager/world)"; exit 2 ;;
esac
TD=/data1/hansol/spoon3537/targets/$V; [ $V = tgt_v32 ] && TD=$C22/streamALL20K/targets/tgt_v32; [ -d $TD/$NM ] || { echo "missing $TD/$NM"; exit 2; }
M=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/$(python3 -c "import json;print(json.load(open('$C22/selection_v29.json'))['$EP'])")/$NM/meshes/$NM
WIN=$(python3 -c "
w=[]
for r in '$RANGES'.split(','):
    a,b=map(float,r.split(':')); t=a
    while t<b-1e-6: e=min(t+${WSTEP:-0.5},b); w.append(f'{t:.2f}:{e:.2f}'); t=e
print(','.join(w))")
OUT=$R/runs/ep$EP/$N; mkdir -p $R/runs/ep$EP
D=""; [ -n "$DEP" ] && D="--dependency=afterok:$DEP"
J1=$(E31_TGT=$TD sbatch --parsable $D -c ${WK:-30} --time ${TL:-02:00:00} -J S3-b-ep$EP-$N $C/job.sbatch wbeamH.py --ep $EP --actor $A --meshes $M --out $OUT --table $TB ${INIT:+--init $INIT} \
  --windows $WIN --beam 4 --n 16 --per-parent 2 --look 1.2 --pre 0.15 --post 0.1 --knots 3 --pos-std 0.004 --rot-std 0.06 --sq-std 0.04 \
  --scales 0.1,0.25,0.5,1.0 --rand-sides right,left --hands ${HANDS:-wooden_spoon:right} ${OW:+--ow $OW} --ilc 0.3,0.6,1.0,1.3 --ilc-sigmas 0.03,0.06 --ilc-modes full,pos,rot,rotanc \
  --tol $TOL --mppe-w 1.0 --workers ${WK:-30} --seed $((1000*EP+${#N})) --hinge $HG --roll wooden_spoon:right --roll-alphas 0.5,1.0,1.5,2.0 --sq 0.1,0.2 \
  --adv 0.5,0.9,1.3 --adv-g 0.5,1.0 --adv-sides right,both)
J2=$(sbatch --parsable --dependency=afterok:$J1 -J S3-sv-ep$EP-$N $C/s3svst.sbatch $V $EP $N $A $OUT/final_0.npz)
TAG=S3_ep${EP}_$N; FCD=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/$FC
J3=$(sbatch --parsable --dependency=afterok:$J2 -p debug --exclude=node200 -c 2 --mem=16G --time=00:40:00 -J S3-post-ep$EP-$N -o $R/logs/%x-%j.log --wrap "
bash $C/s3score.sh $EP $TAG ${TAG}_vs_$V $TD; bash $C/s3score.sh $EP $TAG ${TAG}_vs_v32 $C22/streamALL20K/targets/tgt_v32;
bash $C/s3score.sh $EP $FCD FINAL_ep${EP}_vs_$V $TD; bash $C/s3score.sh $EP $FCD FINAL_ep${EP}_vs_v32 $C22/streamALL20K/targets/tgt_v32;
cd $R/tmp && PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages /home/hansol/v2d/.venvs/v2d-host/bin/python -B $C/s3pkg.py $EP $TAG ${PKGLABEL:-track3_FINALc_ep${EP}${N}_$V} ${BASEP:-}")
echo "ep$EP $N target=$V windows=$WIN beam=$J1 svst=$J2 post=$J3"
echo "$(date +%FT%T) ep$EP $N target=$V beam=$J1 svst=$J2 post=$J3 windows=$WIN" >> $R/jobs.txt
