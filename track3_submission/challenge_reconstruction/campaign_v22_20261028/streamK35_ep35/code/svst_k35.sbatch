#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=K35-svst
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK35_ep35/logs/%x-%j.log
# usage: sbatch svst_k35.sbatch HYP(K35 target label) NAME ACTOR TABLE.npz|none WORLD_EPD
#  save (bake TABLE into ACTOR phase weights; official evaluate_policy_component against WORLD_EPD = K35 world with the hypothesis target in
#  the physics world) -> stageK_k35 (stage_a.sh) -> G18 mesh-only swap (meshes asserted == v29) -> build-v2 (run_one body, TAG K35_<HYP>_<NAME>) -> vs_k35.sh
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; H=$C22/streamK35_ep35; R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
V=$1; N=$2; A=$3; TB=$4; WE=$5; E=35; NM=episode_000035; TAG=K35_${V}_$N
export E31_TGT=$H/targets/$V E31_EPD=$WE OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent TMPDIR=$H/tmp
M=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/G18_ep35_H13b2/$NM/meshes/$NM
SV=$H/runs/save_${V}_$N; B=$H/staging/${V}_$N
TA=""; [ "$TB" != none ] && TA="--table $TB"
[ -f $SV/actor/evaluation/result.json ] || (cd $TMPDIR; $PY -B -u $H/code/h13res.py save --ep $E --actor $A --meshes $M --out $SV $TA 2>&1 | grep -v -E "Warn|warn\(")
NOTE="K35: v29 ep35 actor (H13_ep35_b2) + window-beam phase residuals (table $TB), re-retargeted vs K35 hypothesis $V (ep35 static-window spoon roll = physical rest roll on our stereo table plane)"
bash $H/code/stage_a.sh $E $SV/actor $B "$NOTE"
$PY -B $H/code/g18swap.py $E $B/components/$NM
bash /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/run_one.sbatch $E $B/components/$NM $TAG
bash $H/code/vs_k35.sh $TAG $V
