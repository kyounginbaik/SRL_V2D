#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=A20-svst
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamALL20K/logs/%x-%j.log
# usage: sbatch svst.sbatch V(K33 target label) EP NAME ACTOR SAVER(h13res.py|mres32H13.py|mresH.py) TABLE.npz|none
#  save (bake TABLE into ACTOR phase weights; official evaluate_policy_component) -> unchanged source stager (stage_a.sh / stage_b.sh, + E31 lineage)
#  -> G18 mesh-only swap (g18swap.py; meshes asserted == v29) -> build-v2 (run_one.sbatch body, TAG E31{A,B}_epNN_NAME) -> vs.sh
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; H=$C22/streamALL20K; R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
V=$1; E=$2; N=$3; A=$4; SAVER=$5; TB=$6; NM=$(printf episode_%06d $E); TAG=A20_ep${E}_$N
export E31_TGT=$H/targets/$V OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent TMPDIR=$H/tmp
M=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/$(python3 -c "import json;print(json.load(open('$C22/selection_v29.json'))['$E'])")/$NM/meshes/$NM
SV=$H/runs/save_${V}_ep${E}_$N; B=$H/staging/${V}_ep${E}_$N
TA=""; [ "$TB" != none ] && TA="--table $TB"
[ -f $SV/actor/evaluation/result.json ] || (cd $TMPDIR; $PY -B -u $H/code/$SAVER save --ep $E --actor $A --meshes $M --out $SV $TA 2>&1 | grep -v -E "Warn|warn\(")
NOTE="ALL20-K: v29 actor + window-beam phase residuals (table $TB), tightened vs our own target $V (GT-free)"
case $E in 3|19|8|1|14|25|30|16|10|15) bash $H/code/stage_b.sh $E $SV/actor $B "$NOTE" ;; *) bash $H/code/stage_a.sh $E $SV/actor $B "$NOTE" ;; esac
$PY -B $H/code/g18swap.py $E $B/components/$NM
bash /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/run_one.sbatch $E $B/components/$NM $TAG
bash $H/code/vs.sh $E $TAG $V
