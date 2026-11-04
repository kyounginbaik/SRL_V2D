#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=2-12:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
# Usage: sbatch -J p37 prep.sbatch 37 [seed]   Builds the episode, runs Mink IK, seeded search, hardening and checks each output.
set -uo pipefail
N=$1; SEED=${2:-41}; EP=$(printf recon_%06d $N); ZEP=$(printf episode_%06d $N)
R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; W=/scratch2/exaFLOPs/latest_recon/run37; B=/scratch2/exaFLOPs/hawor_rerun/root; E=$H/episodes/$EP; S=$R/search/$EP; mkdir -p $S; C=/uniretarget_yam/episodes/$EP; TMP=$H/.cache/tmp_p_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
fl() { apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,NVIDIA_DRIVER_CAPABILITIES=all,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}",$ENVX -B "$TMP:/tmp" -B "$H:/uniretarget_yam" -B "$H/flash_chord:/workspace:ro" -B "$H/.cache/warp:/tmp/.cache/warp" -B "$H/.cache/assets:/workspace/src/flash_chord/assets/.cache" -B "$R/comparison:/cmp" -B "$R/search:/search" -B /scratch2 --pwd "$PD" $R/flash_chord_yam.sif "$@"; }
rm -f $E/yam/flash/other_arm.npz
ENVX=X=1; PD=/uniretarget_yam/code
fl python prepare_recon.py --zip_root $B --episode $N --output $C --objects_dir /uniretarget_yam/objects --joint_names_from $C/../episode_000037/yam/flash/fixed/reference.parquet || { echo STEP_FAILED prepare; exit 1; }
META=$E/recon_meta.json; python3 -c "
import json;d=json.load(open('$META'));m=d['movers'];print('movers',m,'table',d['table_height'],'objects',d['objects'])"
read OBJ IDX SIDE <<<$(python3 -c "
import json;m=json.load(open('$META'))['movers'];m=max(m,key=lambda x:x['move_cm']);print(m['object'],m['index'],m['side'])"); echo "mover $OBJ index $IDX side $SIDE"
python3 -c "import json;json.dump(json.load(open('$META'))['up_in_target_world'],open('$E/up.json','w'))"
MAINOBJ=$OBJ; MAINIDX=$IDX
read IDX SIDE <<<$(python3 -c "
import json;h=json.load(open('$META'))['holds'][0];print(h['index'],h['side'])"); OBJ=$(python3 -c "
import json;d=json.load(open('$META'));print(d['objects'][d['holds'][0]['index']]['name'])")
read NST SIDES <<<$(python3 -c "
import json;h=json.load(open('$META'))['holds'];print(len(h),','.join(sorted({x['side'] for x in h})))"); python3 -c "
import json;d=json.load(open('$META'));print('schedule',d['hold']['kinds'],[(x['object'],x['side'],x['start'],x['end'],x.get('release')) for x in d['holds']],'frames',d['frames'],'cover',d['hold']['coverage'])"
if [ "$NST" -gt 1 ]; then PLANS="--plan_sides $SIDES"; STARTONLY=--start-only; S2=multi; else PLANS=""; STARTONLY=""; S2=""; fi
ENVX=X=1; PD=$W; read FR PAD <<<$(python3 -c "import json;d=json.load(open('$META'));print(d['frames'],d.get('pad',0))")
python3 -c "
import json;d=json.load(open('$META'));print(' '.join(sorted({str(h['index'])+':'+h['object'] for h in d['holds'] if h['index']!=$MAINIDX})))" > $E/other_bundles.txt
for X in $(cat $E/other_bundles.txt); do KK=${X%%:*}; NN=${X#*:}; fl python $W/convert_hawor.py --frames $FR --pad $PAD --hawor $B/hands/$ZEP/hawor_${NN}_result.npz --target $B/targets/$ZEP/${NN}_trajectory.npz --up $E/up.json --side $SIDE --output $E/bundle_$KK | tail -1 | cut -c1-100 > /dev/null || echo "bundle $NN failed"; done
ENVX=X=1; PD=$W; read FR PAD <<<$(python3 -c "import json;d=json.load(open('$META'));print(d['frames'],d.get('pad',0))"); fl python $W/convert_hawor.py --frames $FR --pad $PAD --hawor $B/hands/$ZEP/hawor_${MAINOBJ}_result.npz --target $B/targets/$ZEP/${MAINOBJ}_trajectory.npz --up $E/up.json --side $SIDE --output $E/bundle | tail -1 | cut -c1-600 || { echo STEP_FAILED convert; exit 1; }
ENVX=X=1; PD=/uniretarget_yam/code; fl python place_check.py $STARTONLY --episode $C --side $SIDE --object $IDX --output $C/place_allowed.json 2>&1 | grep -E 'allowed|Traceback' | cut -c1-200
cd $W; apptainer exec --containall --no-home --home /tmp --env OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=$W/rg/source/robotic_grounding:$W/rg/lib:$W/pydeps,OMNI_KIT_ACCEPT_EULA=YES -B /scratch2 -B $TMP:/tmp --pwd $W /lustre/exaflops/singularity_exports/robotic-grounding-mink-1.3.0.sif /isaac-sim/python.sh yam_mink.py --bundle $E/bundle --flash $E/yam/flash --source source --side $SIDE --object_index $MAINIDX --output $E/yam/flash/mink --allowed $E/place_allowed.json $PLANS --plan_out $E/place.json 2>&1 | grep -E 'plan_offset|Traceback|Error' | cut -c1-220
rm -f $E/yam/flash/mink/reference.parquet
[ -f $E/place.json ] && { ENVX=X=1; PD=/uniretarget_yam/code; fl python place_grasp.py --episode $C --place $C/place.json --workers 30 2>&1 | grep -E 'place_grasp|Traceback|Error' | cut -c1-360; fl python prepare_recon.py --zip_root $B --episode $N --output $C --objects_dir /uniretarget_yam/objects --joint_names_from $C/../episode_000037/yam/flash/fixed/reference.parquet --offset $C/place.json > /dev/null 2>&1 || { echo STEP_FAILED replace; exit 1; }; }
cd $W; apptainer exec --containall --no-home --home /tmp --env OMP_NUM_THREADS=8,PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,PYTHONPATH=$W/rg/source/robotic_grounding:$W/rg/lib:$W/pydeps,OMNI_KIT_ACCEPT_EULA=YES -B /scratch2 -B $TMP:/tmp --pwd $W /lustre/exaflops/singularity_exports/robotic-grounding-mink-1.3.0.sif /isaac-sim/python.sh yam_mink.py --bundle $E/bundle --flash $E/yam/flash --source source --side $SIDE --object_index $MAINIDX --output $E/yam/flash/mink 2>&1 | grep -E 'hand_repair|reference_contact|^frame (0|50|100)|frames"|spoon fit|YAM_IK_DONE|Traceback|Error' | cut -c1-700
[ -f $E/yam/flash/mink/reference.parquet ] || { echo STEP_FAILED mink; exit 1; }
[ -z "$S2" ] && ENVX=X=1 PD=/uniretarget_yam/code && fl python other_arm.py --episode $C --side $SIDE --object $IDX --parquet $C/yam/flash/mink/reference.parquet 2>&1 | grep -E 'other_arm|Traceback|Error' | cut -c1-300
[ -n "${STOP_AFTER_MINK:-}" ] && { echo MINK_ONLY_DONE; exit 0; }
ENVX=X=1; PD=/uniretarget_yam/code; fl python place_check.py $STARTONLY --episode $C --side $SIDE --object $IDX --report 2>&1 | grep -E 'start_clear|Traceback' | cut -c1-500
ENVX=X=1; PD=/uniretarget_yam/code
: > $S/pool.txt
rm -f $S/stage_current.json
read _K _S E1 FUTS _M _N FRZ <<<$(python3 /scratch2/exaFLOPs/srl_yam_run/pipe/stage.py $META 0 "" /dev/null)
FUT1=""; [ "$FUTS" != "-" ] && FUT1="--future ${FUTS//,/ }"; IGN=""; [ "$FRZ" != "-" ] && IGN="--ignored ${FRZ//,/ }"; FUT1="$FUT1 $IGN"
for SD in $SEED $((SEED+1)) $((SEED+2)); do
  ENVX=X=1; PD=/uniretarget_yam/code
  fl python yam_refine.py --reference $C/yam/flash/mink/reference.parquet --episode $C --side $SIDE --object $IDX --output /search/$EP/s$SD --end $E1 $FUT1 --workers 30 --proposals 1200 --population 300 --stall 40 --seed $SD --onset-floor 1.4 2>&1 | grep -E '"stage"|mean_tip_cm|Traceback|Error' | grep -v '"generation"' | cut -c1-420
  [ -f $S/s$SD/pass.json ] || { echo "seed $SD: no grasp from search"; continue; }
  ENVX=YAM_END=$E1; fl python yam_harden.py --episode $C --side $SIDE --object $IDX --tape /search/$EP/s$SD/pass.json --output /search/$EP/s$SD/harden --levels 0.75 1.0 1.5 --workers 30 --onset-floor 1.4 2>&1 | grep -E '"report": "hardened"|Traceback' | cut -c1-140
  ENVX=X=1
  for V in harden/hardened pass; do [ -f $S/s$SD/$V.json ] || continue; NAME=$(echo $V | tr '/' '_'); fl python yam_flash_ref.py --episode $C --side $SIDE --object $IDX --source $C/yam/flash/source/reference.parquet --tape /search/$EP/s$SD/$V.json --output /search/$EP/s$SD/flash_$NAME --onset-floor 1.4 2>&1 | tail -1 | cut -c1-160
    P=/search/$EP/s$SD/flash_$NAME/reference.parquet; PD=/workspace; ENVX=RECON_HOLDS_UPTO=1,RECON_EP=$C,RECON_SIDE=$SIDE,RECON_OBJ=$IDX,RECON_PARQUET=$P; fl python /cmp/ref_check.py A $P 2>&1 | grep -E '"zero_action"|Traceback' > $S/s$SD/check_$NAME.txt || true; ENVX=X=1; PD=/uniretarget_yam/code
    python3 - "$S/s$SD/check_$NAME.txt" "$P" "$S/pool.txt" "$NAME" <<'PY'
import sys,json
good=lambda x:sum(v for k,v in x['kinds'].items() if k in('held','off_track'))/sum(x['kinds'].values())>=.9 and x.get('lift',0)>=.8
f,p,pool,name=sys.argv[1:];r=[json.loads(l) for l in open(f) if l.startswith('{')]
if not r:print(name,'playback produced nothing');sys.exit()
mn=min(x['passed'] for x in r);print(name,'flash_'+name,'playback',r[0]['kinds'],'carried',all(good(x) for x in r),'lift',round(r[0].get('lift',0),2),'pass',mn,'cm',round(r[0]['cm'],2),'deg',round(r[0]['deg'],1),'rise',round(r[0]['rise'],1),flush=True)
if all(good(x) for x in r):open(pool,'a').write(p+'\n')
PY
  done
  [ -s $S/pool.txt ] && { echo "seed $SD: grasp holds in Newton"; break; }
done
[ -s $S/pool.txt ] || { echo STEP_FAILED newton; exit 1; }
I=1
while [ "$I" -lt "$NST" ]; do
  BP=$(head -1 $S/pool.txt); BC=$(dirname $BP)/ctrl.npz; ST=$((I+1)); J=$S/stage${ST}_join.json; HOSTBP=${BP/#\/search/$R\/search}
  read KI SI EI FUTS MID _N FRZ <<<$(python3 /scratch2/exaFLOPs/srl_yam_run/pipe/stage.py $META $I "$HOSTBP" $J)
  python3 - "$J" "$R/search" <<'PY'
import json,sys
j,r=sys.argv[1:];d=json.load(open(j))
if d.get('parquet'):d['parquet']=d['parquet'].replace(r,'/search')
json.dump(d,open(j,'w'))
PY
  CJ=/search/$EP/stage${ST}_join.json; FUTI=""; [ "$FUTS" != "-" ] && FUTI="--future ${FUTS//,/ }"; [ "$FRZ" != "-" ] && FUTI="$FUTI --ignored ${FRZ//,/ }"
  echo "{\"stage\": $ST, \"join\": \"$CJ\", \"base_ctrl\": \"$BC\"}" > $S/stage_current.json
  echo "stage $ST of $NST: object $KI side $SI to frame $EI midair $MID"; : > $S/pool.txt
  for SD in $SEED $((SEED+1)) $((SEED+2)); do
    ENVX=X=1; PD=/uniretarget_yam/code; O=/search/$EP/stage${ST}_s$SD
    fl python yam_refine.py --reference $C/yam/flash/mink/reference.parquet --episode $C --side $SI --object $KI --output $O --base-tape $BC --join $CJ --end $EI $FUTI --workers 30 --proposals 1200 --population 300 --stall 40 --seed $SD --onset-floor 1.4 2>&1 | grep -E '"stage"|mean_tip_cm|Traceback|Error' | grep -v '"generation"' | cut -c1-420
    [ -f $S/stage${ST}_s$SD/pass.json ] || { echo "stage $ST seed $SD: no grasp from search"; continue; }
    fl python yam_flash_ref.py --join $CJ --episode $C --side $SI --object $KI --source $C/yam/flash/source/reference.parquet --tape $O/pass.json --base-tape $BC --base-parquet $BP --output $O/flash_pass --onset-floor 1.4 2>&1 | tail -1 | cut -c1-160
    P=$O/flash_pass/reference.parquet; PD=/workspace; ENVX=RECON_HOLDS_UPTO=$ST,RECON_EP=$C,RECON_SIDE=$SIDE,RECON_OBJ=$IDX,RECON_PARQUET=$P; fl python /cmp/ref_check.py A $P 2>&1 | grep -E '"zero_action"|Traceback' > $S/stage${ST}_s$SD/check.txt || true; ENVX=X=1; PD=/uniretarget_yam/code
    python3 - "$S/stage${ST}_s$SD/check.txt" "$P" "$S/pool.txt" "stage$ST" <<'PY'
import sys,json
good=lambda x:sum(v for k,v in x['kinds'].items() if k in('held','off_track'))/sum(x['kinds'].values())>=.9 and x.get('lift',0)>=.8
f,p,pool,name=sys.argv[1:];r=[json.loads(l) for l in open(f) if l.startswith('{')]
if not r:print(name,'playback produced nothing');sys.exit()
mn=min(x['passed'] for x in r);print(name,'playback',r[0]['kinds'],'carried',all(good(x) for x in r),'lift',round(r[0].get('lift',0),2),'pass',mn,'cm',round(r[0]['cm'],2),'per object',r[0].get('holds'),flush=True)
if all(good(x) for x in r):open(pool,'a').write(p+'\n')
PY
    [ -s $S/pool.txt ] && { echo "stage $ST: holds in Newton, seed $SD"; break; }
  done
  [ -s $S/pool.txt ] || { echo STEP_FAILED stage_$ST; exit 1; }
  I=$((I+1))
done
PD=/workspace; P=$C/yam/flash/mink/reference.parquet; ENVX=RECON_EP=$C,RECON_SIDE=$SIDE,RECON_OBJ=$IDX,RECON_PARQUET=$P; fl python /cmp/ref_check.py A $P 2>&1 | grep -E '"zero_action"' | head -1 | cut -c1-260
echo PREP_DONE $N
