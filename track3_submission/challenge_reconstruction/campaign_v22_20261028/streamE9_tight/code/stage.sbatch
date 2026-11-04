#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=E9-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE9_tight/logs/stage-%j.log
# usage: sbatch stage.sbatch EP ACTOR_DIR BASE "selection note" 'objects json'
# Unchanged stage scripts of the streams that built each episode world (same calls as N8 stage.sbatch (eps 1/14), U8 stage15/16.sbatch) + E9 lineage.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; E9=$C22/streamE9_tight; U8=$C22/streamU8_15_16; SN=$C22/streamN8_1_8_10_14_20_30
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
EP=$1; A=$2; B=$3; NOTE=${4:-"E9 window beam search on the phase weights"}; OBJ=${5:-'{}'}
DL="--lineage-file $E9/code/mresE9.py --lineage-file $E9/code/mlibE9.py --lineage-file $E9/code/wbeamE9.py --lineage-file $C22/streamD9_tight/code/wbeam9.py"
cd /tmp
case $EP in
  1)  $PY -B -u $SN/code/stage_n8.py --episode 1 --distilled $A --base $B --teacher $SN/ep1/kcem/k2_lr_r45 $DL ;;
  14) $PY -B -u $SN/code/stage_n8.py --episode 14 --distilled $A --base $B --teacher $SN/ep14/kcem/k_l_10_65 $DL ;;
  15) $PY -B $E9/code/mkbaseE9.py $U8/staging/ep15_c $B $A "$NOTE" "$OBJ"
      WORLD=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamS_ep15/world_6prong_nt
      source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamS_ep15/code/env.sh
      $PY -u $N/code/stage_seven_component_newtrack.py --episode 15 --base $B --actor-report $B/actor_report.json --lineage-file $WORLD/lineage.json --lineage-file $WORLD/world_meshes.json --lineage-file $WORLD/world_check.json --lineage-file $N/code/build_world.py --lineage-file $N/code/port_specs.py --lineage-file $U8/code/mres.py --lineage-file $U8/code/mlib.py --lineage-file $U8/code/wbeam_u8.py $DL ;;
  16) $PY -B $E9/code/mkbaseE9.py $U8/staging/ep16_b $B $A "$NOTE" "$OBJ"
      WORLD=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep16/world_T
      source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/env.sh
      F=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep16/code; REC=$WORLD/source/episode_000016/target_retarget.json
      $PY -u $F/stage_seven_component_retargeted.py --episode 16 --base $B --actor-report $B/actor_report.json --lineage-file $WORLD/lineage.json --lineage-file $WORLD/world_meshes.json --lineage-file $WORLD/world_check.json --lineage-file $N/code/build_world_v2.py --lineage-file $F/build_world_f16.py --lineage-file $F/tlib16.py --lineage-file $REC --lineage-file $U8/code/mres.py --lineage-file $U8/code/mlib.py --lineage-file $U8/code/wbeam_u8.py $DL ;;
esac
echo STAGE_DONE
