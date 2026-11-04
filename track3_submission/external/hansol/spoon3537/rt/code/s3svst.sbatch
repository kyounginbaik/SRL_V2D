#!/bin/bash
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=S3-svst
#SBATCH --output=/data1/hansol/spoon3537/rt/logs/%x-%j.log
# [spoon3537 copy of streamALL20K svst.sbatch] save (bake TABLE into ACTOR, mresH) -> unchanged stager (stage_s3.sh = stage_a.sh, TMPDIR moved)
# -> G18 mesh-only swap -> build-v2 (s3run_one.sh) -> exactness check. Everything written under /data1/hansol/spoon3537/rt.
# usage: sbatch s3svst.sbatch TARGET_LABEL EP NAME ACTOR TABLE.npz|none
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; H=/data1/hansol/spoon3537/rt; R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=/home/hansol/v2d/.venvs/v2d-host/bin/python
V=$1; E=$2; N=$3; A=$4; TB=$5; NM=$(printf episode_%06d $E); TAG=S3_ep${E}_$N
TD=/data1/hansol/spoon3537/targets/$V; [ $V = tgt_v32 ] && TD=$C22/streamALL20K/targets/tgt_v32
export E31_TGT=$TD OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent:/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages TMPDIR=$H/tmp
M=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/$(python3 -c "import json;print(json.load(open('$C22/selection_v29.json'))['$E'])")/$NM/meshes/$NM
SV=$H/runs/save_${V}_ep${E}_$N; B=$H/staging/${V}_ep${E}_$N
TA=""; [ "$TB" != none ] && TA="--table $TB"
[ -f $SV/actor/evaluation/result.json ] || (cd $TMPDIR; $PY -B -u $H/code/mresH.py save --ep $E --actor $A --meshes $M --out $SV $TA 2>&1 | grep -v -E "Warn|warn\(")
NOTE="ALL20-K: v29 actor + window-beam phase residuals (table $TB), tightened vs our own target $V (GT-free)"
rm -rf $B
case $E in 3|19|8|1|14|25|30|16|10|15) bash $H/code/stage_b_s3.sh $E $SV/actor $B "$NOTE" ;; *) bash $H/code/stage_s3.sh $E $SV/actor $B "$NOTE" ;; esac
/home/hansol/v2d/.venvs/v2d-host/bin/python -B $H/code/g18swap_s3.py $E $B/components/$NM
rm -rf $H/candidates/$TAG
bash $H/code/s3run_one.sh $E $B/components/$NM $TAG
/home/hansol/v2d/.venvs/v2d-host/bin/python -c "import json;d=json.load(open('$H/candidates/$TAG/$NM/completed.json'));print('BUILD exact_previous_physical_states_actions =', d.get('exact_previous_physical_states_actions'), 'checkpoint', d['checkpoint']['sha256'] if isinstance(d.get('checkpoint'),dict) else d.get('checkpoint'))"
echo SVST_DONE $TAG
