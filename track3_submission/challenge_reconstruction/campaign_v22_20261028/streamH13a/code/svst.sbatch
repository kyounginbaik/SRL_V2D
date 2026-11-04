#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=H13a-svst
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13a/logs/%x-%j.log
# usage: sbatch svst.sbatch EP NAME ADV [TABLE.npz|none]
#   save (bake ADV time-advance + TABLE into the v27 actor; official evaluate_policy_component) -> stage -> build-v2 (run_one.sbatch body, TAG H13_epNN_NAME)
#   -> exactness + score_any vs target_final_v12b_g15.
set -euo pipefail
source /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH13a/code/env.sh
E=$1; N=$2; ADV=$3; TB=${4:-none}; TAG=H13_ep$(printf %02d $E)_$N; NM=$(printf episode_%06d $E)
eval A=\$A$E; eval M=\$M$E
RS=$H/code/h13res.py; [ $E = 32 ] && RS=$H/code/mres32H13.py
TA=""; [ "$TB" != none ] && TA="--table $TB"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
SV=$H/runs/save_ep${E}_$N
[ -f $SV/actor/evaluation/result.json ] || (cd $H/tmp; H13_ADV="$ADV" $PY -B -u $RS save --ep $E --actor $A --meshes $M --out $SV $TA 2>&1 | grep -v -E "Warn|warn\(")
bash $H/code/stage.sbatch $E $SV/actor $H/staging/ep${E}_$N "H13a time-advance $ADV frames + table $TB (window beam vs target_final_v12b_g15)"
bash /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/run_one.sbatch $E $H/staging/ep${E}_$N/components/$NM $TAG
bash $H/code/vs.sh $E $TAG
