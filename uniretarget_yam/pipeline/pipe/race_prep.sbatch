#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch2/exaFLOPs/srl_yam_run/pipe/%x-%j.log
# Usage: sbatch -J rp5 race_prep.sbatch N "name:dir:index ..."   Turns top.json entries from the last stage into references and checks each in Newton with no policy.
set -uo pipefail
N=$1; EP=$(printf recon_%06d $N); R=/scratch2/exaFLOPs/srl_yam_run; H=/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam; S=$R/search/$EP; C=/uniretarget_yam/episodes/$EP; META=$H/episodes/$EP/recon_meta.json; TMP=$H/.cache/tmp_r_${SLURM_JOB_ID}; mkdir -p $TMP; trap 'rm -rf $TMP' EXIT
fl() { apptainer exec --nv --containall --no-home --home /tmp --env PYTHONUNBUFFERED=1,PYTHONDONTWRITEBYTECODE=1,OMP_NUM_THREADS=1,NVIDIA_DRIVER_CAPABILITIES=all,CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}",$ENVX -B "$TMP:/tmp" -B "$H:/uniretarget_yam" -B "$H/flash_chord:/workspace:ro" -B "$H/.cache/warp:/tmp/.cache/warp" -B "$H/.cache/assets:/workspace/src/flash_chord/assets/.cache" -B "$R/comparison:/cmp" -B "$R/search:/search" -B /scratch2 --pwd "$PD" $R/flash_chord_yam.sif "$@"; }
read KI SI NST <<<$(python3 -c "import json;h=json.load(open('$META'))['holds'];print(h[-1]['index'],h[-1]['side'],len(h))")
read IDX SIDE <<<$(python3 -c "import json;h=json.load(open('$META'))['holds'][0];print(h['index'],h['side'])")
BC=$(python3 -c "import json;print(json.load(open('$S/stage_current.json'))['base_ctrl'])"); BP=$(dirname $BC)/reference.parquet; CJ=/search/$EP/stage${NST}_join.json
for X in $2; do IFS=: read G D I <<<"$X"; O=/search/$EP/race/$G; mkdir -p $S/race/$G
  python3 -c "import json;json.dump(json.load(open('$S/$D/top.json'))[$I],open('$S/race/$G/tape.json','w'))"
  ENVX=X=1; PD=/uniretarget_yam/code; fl python yam_flash_ref.py --join $CJ --episode $C --side $SI --object $KI --source $C/yam/flash/source/reference.parquet --tape $O/tape.json --base-tape $BC --base-parquet $BP --output $O/flash --onset-floor 1.4 2>&1 | tail -1 | cut -c1-160
  P=$O/flash/reference.parquet; [ -f $S/race/$G/flash/reference.parquet ] || { echo "$G NO_REFERENCE"; continue; }
  ENVX=RECON_HOLDS_UPTO=$NST,RECON_EP=$C,RECON_SIDE=$SIDE,RECON_OBJ=$IDX,RECON_PARQUET=$P; PD=/workspace; fl python /cmp/ref_check.py A $P 2>&1 | grep -E '"zero_action"|Traceback' > $S/race/$G/check.txt
  python3 - "$G" "$S/race/$G/check.txt" <<'PY'
import sys,json
r=[json.loads(l) for l in open(sys.argv[2]) if l.startswith('{')]
print(sys.argv[1],'NEWTON',[{k:(round(v,2) if isinstance(v,float) else v) for k,v in x.items() if k in('kinds','lift','frontier','cm','deg')} for x in r] or 'nothing')
PY
done
echo RACE_PREP_DONE
