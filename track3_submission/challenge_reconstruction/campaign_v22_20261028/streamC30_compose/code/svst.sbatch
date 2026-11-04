#!/bin/bash
#SBATCH -p debug --nodelist=node100 -c 2 --mem=16G -t 00:50:00 -J C30-svst -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC30_compose/logs/svst_%j.log
# usage: C30_TGT=<target> sbatch --export=ALL svst.sbatch LIB(a|b) EP ACTOR TABLE.npz TAG
#  save (bake TABLE into ACTOR's phase weights + official evaluate_policy_component; H13a h13res / H13b mresH),
#  stage with the unchanged source-stream stagers (C30 copies of H13a stage.sbatch / H13b stageH.sbatch),
#  spoon eps: + G18 mesh-only swap (v29 spoon mesh), build-v2 (common_v2_candidates/run_one.sbatch), score_any vs C30_TGT and v14_g17.
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; X=$C22/streamC30_compose; B=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
L=$1; E=$2; A=$3; TB=$4; TAG=$5; NM=$(printf episode_%06d $E); SV=$X/runs/save_$TAG; ST=$X/staging/$TAG; mkdir -p $X/tmp $X/staging
echo "C30_TGT=$C30_TGT"; [ -e $B/$TAG ] && { echo "exists $B/$TAG"; exit 1; }
if [ $L = a ]; then
  source $X/h13a/code/env.sh; eval M=\$M$E; RS=$X/h13a/code/h13res.py; [ $E = 32 ] && RS=$X/h13a/code/mres32H13.py
  [ -f $SV/actor/evaluation/result.json ] || (cd $X/tmp; unset H13_ADV; $PY -B -u $RS save --ep $E --actor $A --meshes $M --out $SV --table $TB 2>&1 | grep -v -E "Warn|warn\(")
  bash $X/h13a/code/stage.sbatch $E $SV/actor $ST "C30: v29 actor + window-beam phase residuals vs $(basename $C30_TGT)"
else
  source $X/h13b/code/env.sh; eval S=\$S$E; M=$(ls -d $S/components/episode_*/meshes/episode_*)
  [ -f $SV/actor/evaluation/result.json ] || (cd $X/tmp; $PY -B -u $X/h13b/code/mresH.py save --ep $E --actor $A --meshes $M --table $TB --out $SV 2>&1 | grep -v -E "Warn|warn\(")
  bash $X/h13b/code/stageH.sbatch $E $SV/actor $ST "C30: v29 actor + window-beam phase residuals vs $(basename $C30_TGT)" '{}'
fi
COMP=$ST/components/$NM
case $E in 3|5|6|15|16|32|33|35|36|37)
  $PY -B $X/code/g18swap.py $E $ST ${ST}_g18; COMP=${ST}_g18/components/$NM ;;
esac
bash $B/run_one.sbatch $E $COMP $TAG
python3 -c "import json;d=json.load(open('$B/$TAG/$NM/completed.json'));print('EXACT',d['exact_previous_physical_states_actions'])"
bash $C22/code/score_any.sh $E $B/$TAG/$NM $TAG $C30_TGT >/dev/null 2>&1; cat $C22/scores/cand/$TAG/score.json
bash $C22/code/score_any.sh $E $B/$TAG/$NM ${TAG}_vs_v14 $C22/target_final_v14_g17 >/dev/null 2>&1; cat $C22/scores/cand/${TAG}_vs_v14/score.json
echo SVST_DONE
