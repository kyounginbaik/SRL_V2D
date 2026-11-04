#!/bin/bash
#SBATCH -p debug --nodelist=node100 -c 2 --mem=16G -t 00:50:00 -J G19-kobuild -o /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate/logs/kobuild_%j.log
# usage: kobuild.sbatch a|b EP SHIFT NAME
# Deliberate wrong-phase knockout: the episode's v27 own-target actor (H13a/H13b env.sh) with its phase reference shifted by SHIFT frames
# (negative = delayed), saved with the unchanged H13a h13res.py save / H13b tadv.py --save (official evaluate_policy_component rollout),
# staged with the unchanged H13a stage.sbatch / H13b stageH.sbatch, converted with common_v2_candidates/run_one.sbatch (TAG G19KO_epNN_NAME).
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
S=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamG19_v29locate; C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028
W=$1; E=$2; SH=$3; N=$4; NM=$(printf episode_%06d $E); TAG=KO19_ep$(printf %02d $E)_$N
SV=$S/ko1/runs/ep${E}_$N; BASE=$S/ko1/staging/ep${E}_${N}_s2; mkdir -p $S/tmp; cd $S/tmp
NOTE="G19 diagnostic knockout: own-target v27 actor with its phase reference shifted by $SH frames (wrong phase); deliberately fails MP-SR; not a leaderboard component"
if [ $W = a ]; then source $C22/streamH13a/code/env.sh; eval A=\$A$E; eval M=\$M$E
  RS=$C22/streamH13a/code/h13res.py; [ $E = 32 ] && RS=$C22/streamH13a/code/mres32H13.py
  [ -f $SV/actor/evaluation/result.json ] || H13_ADV="$SH" $PY -B -u $RS save --ep $E --actor $A --meshes $M --out $SV 2>&1 | grep -v -E "Warn|warn\(" || true
  bash $S/code/kostage_a.sbatch $E $SV/actor $BASE "$NOTE"
else source $C22/streamH13b/code/env.sh; eval A=\$A$E; eval SS=\$S$E; M=$(ls -d $SS/components/episode_*/meshes/episode_*)
  [ -f $SV/actor/evaluation/result.json ] || $PY -B -u $C22/streamH13b/code/tadv.py --ep $E --actor $A --meshes $M --out $SV --save=$SH 2>&1 | grep -v -E "Warn|warn\(" || true
  bash $S/code/kostage_b.sbatch $E $SV/actor $BASE "$NOTE" '{}'
fi
bash /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/run_one.sbatch $E $BASE/components/$NM $TAG
python3 -c "import json;d=json.load(open('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/$TAG/$NM/completed.json'));print('EXACT',d['exact_previous_physical_states_actions'])"
