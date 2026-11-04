#!/bin/bash







# usage: sbatch stageH.sbatch EP ACTOR_DIR BASE "note" ['objects json' (eps 15/16)]
# H13b: the same calls as each v27 component's source stream stage script (D9 eps 19/25, H10 code3 ep3, E9 eps 1/14/15/16, F9 eps 8/30,
# X8 stage10 ep10), unchanged stagers, + H13b lineage (mresH/mlibH/wbeamH/tadv). Mesh swaps (ep25 A9 boards, ep30 A9 sweeper) as in D9/F9.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; H13=$C22/streamH13b
D9=$C22/streamD9_tight; E9=$C22/streamE9_tight; F9=$C22/streamF9_tight; H10=$C22/streamH10_v8; L8=$C22/streamL8_3_5_6; Q8=$C22/streamQ8_3_5
U8=$C22/streamU8_15_16; SN=$C22/streamN8_1_8_10_14_20_30; N8=$SN; X=$C22/streamX8_ep10
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
E=$1; A=$2; B=$3; NOTE=${4:-"H13b re-retarget onto target_final_v12b_g15 (time-advance / window beam on the phase weights)"}; OBJ=${5:-'{}'}
HL="--lineage-file $H13/code/mresH.py --lineage-file $H13/code/mlibH.py --lineage-file $H13/code/wbeamH.py --lineage-file $H13/code/tadv.py --lineage-file /data1/hansol/retarget33/code/h13lib.py --lineage-file /data1/hansol/retarget33/code/mlibH.py --lineage-file /data1/hansol/retarget33/code/h13res.py --lineage-file /data1/hansol/retarget33/code/mresH.py --lineage-file /data1/hansol/retarget33/code/mres32H13.py --lineage-file /data1/hansol/retarget33/code/wbeamH.py --lineage-file /data1/hansol/retarget33/code/wbeamH13.py --lineage-file /data1/hansol/retarget33/code/c9beamH13.py --lineage-file /data1/hansol/retarget33/code/wbeam4H13.py --lineage-file /data1/hansol/retarget33/code/wbeamH14.py"
cd /tmp
case $E in
  3)
    DL="--lineage-file $H10/code3/mres9.py --lineage-file $H10/code3/mlib9.py --lineage-file $H10/code3/wbeam9.py --lineage-file $D9/code/mres9.py --lineage-file $D9/code/mlib9.py --lineage-file $D9/code/wbeam9.py --lineage-file $C22/streamU8_15_16/code/mres.py $HL"
    $PY -B -u $H10/code3/stage3.py --episode 3 --distilled $A --base $B --teacher $C22/streamA_ep35/ep3/ret/z6s4_r2.5_s21 \
      --lineage-file $L8/code/l8res.py --lineage-file $L8/code/l8rel.py --lineage-file $L8/code/plib.py --lineage-file $Q8/code/boost.py --lineage-file $Q8/code/wlot.py --lineage-file $Q8/code/q8ilc.py --lineage-file $Q8/code/offgrid.py $DL ;;
  19)
    DL="--lineage-file $D9/code/mres9.py --lineage-file $D9/code/mlib9.py --lineage-file $D9/code/wbeam9.py --lineage-file $C22/streamU8_15_16/code/mres.py $HL"
    source $C22/streamJ8_19_25_28_38/ep19/code/env.sh; W=$J/world_a40; T=$J/cem/a40_kA_LR_2_5.5
    $PY -B -u $J/code/stage_ep19J.py --episode 19 --episode-dir $W/episode_target/episode_000019 --distilled $A --base $B --teacher $T \
      --teacher-note "open-loop servo reference teacher: stream F F_ep19_twin3 teacher (C_ep19_ilc lineage, twin port) re-ported / knot-CEM refined by J8 in $W; actor = J8 a40m2e chain + T8 rigid-hold ILC / CEM residuals + D9 window beam search on the wrist phase weights + H13b time-advance/window beam vs target_final_v12b_g15; teacher only, not shipped" \
      --lineage-file $J/code/port19J.py --lineage-file $J/code/build_world19J.py --lineage-file $J/code/mk_alpha.py --lineage-file $J/code/topple.py --lineage-file $J/code/kcem19.py \
      --lineage-file $J/sweeper_alpha.npz --lineage-file $J/sweeper_P.npz --lineage-file $F/code/common19F.py --lineage-file $F/twin_poses.npz --lineage-file $W/scene/scene_change.json \
      --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py --lineage-file $J/code/actor_res19.py --lineage-file $J/code/merge_actor.py \
      --lineage-file $C22/streamT8_ep19/code/ilc19t.py --lineage-file $C22/streamT8_ep19/code/combo.py $DL ;;
  25)
    DL="--lineage-file $D9/code/mres9.py --lineage-file $D9/code/mlib9.py --lineage-file $D9/code/wbeam9.py --lineage-file $C22/streamU8_15_16/code/mres.py $HL"
    Q=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25; EE=$C22/streamE_ep25_30; W=$EE/ep25/world_T1; J=$C22/streamJ8_19_25_28_38/ep25
    $PY -B $D9/code/mkbase25.py $A $B "$NOTE"
    $PY -u $Q/code/stage_seven_component_corrected_boards.py --episode 25 --base $B --actor-report $B/actor_report.json --lineage-file $W/lineage.json --lineage-file $W/source_corrected/episode_000025/board_track_correction.json --lineage-file $EE/code25/build_worldE.py --lineage-file $EE/code/conv25.py --lineage-file $EE/code/elib.py --lineage-file $Q/code/make_slotted_mesh_C.py --lineage-file $Q/code/cut_slot_body.py --lineage-file $Q/meshes_slotted_C/slotted_mesh_report.json --lineage-file $J/code/res25.py $DL
    $PY -B $D9/code/d9_meshswap25.py $B/components/episode_000025 ;;
  1|14|15|16)
    DL="--lineage-file $E9/code/mresE9.py --lineage-file $E9/code/mlibE9.py --lineage-file $E9/code/wbeamE9.py --lineage-file $D9/code/wbeam9.py $HL"
    case $E in
      1)  $PY -B -u $SN/code/stage_n8.py --episode 1 --distilled $A --base $B --teacher $SN/ep1/kcem/k2_lr_r45 $DL ;;
      14) $PY -B -u $SN/code/stage_n8.py --episode 14 --distilled $A --base $B --teacher $SN/ep14/kcem/k_l_10_65 $DL ;;
      15) $PY -B $H13/code/mkbaseH.py $U8/staging/ep15_c $B $A "$NOTE" "$OBJ"
          WORLD=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamS_ep15/world_6prong_nt
          source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamS_ep15/code/env.sh
          $PY -u $N/code/stage_seven_component_newtrack.py --episode 15 --base $B --actor-report $B/actor_report.json --lineage-file $WORLD/lineage.json --lineage-file $WORLD/world_meshes.json --lineage-file $WORLD/world_check.json --lineage-file $N/code/build_world.py --lineage-file $N/code/port_specs.py --lineage-file $U8/code/mres.py --lineage-file $U8/code/mlib.py --lineage-file $U8/code/wbeam_u8.py $DL ;;
      16) $PY -B $H13/code/mkbaseH.py $U8/staging/ep16_b $B $A "$NOTE" "$OBJ"
          WORLD=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep16/world_T
          source /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamN_ep16/code/env.sh
          F=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF_margins/ep16/code; REC=$WORLD/source/episode_000016/target_retarget.json
          $PY -u $F/stage_seven_component_retargeted.py --episode 16 --base $B --actor-report $B/actor_report.json --lineage-file $WORLD/lineage.json --lineage-file $WORLD/world_meshes.json --lineage-file $WORLD/world_check.json --lineage-file $N/code/build_world_v2.py --lineage-file $F/build_world_f16.py --lineage-file $F/tlib16.py --lineage-file $REC --lineage-file $U8/code/mres.py --lineage-file $U8/code/mlib.py --lineage-file $U8/code/wbeam_u8.py $DL ;;
    esac ;;
  8)
    FL="--lineage-file $F9/code/mresF9.py --lineage-file $F9/code/mlibF9.py --lineage-file $F9/code/wbeamF9.py --lineage-file $C22/streamD9_tight/code/wbeam9.py --lineage-file $C22/streamU8_15_16/code/mres.py $HL"
    $PY -B -u $N8/code/stage_n8.py --episode 8 --distilled $A --base $B --teacher $N8/ep8/kcem/k_lr_20_105 \
      --teacher-note "open-loop servo reference: v21C teacher ported into the target_final_v5-referenced world (port_n8.py) and re-planned (knot CEM kcem_n8.py) by stream N8; actor = N8_ep08_klr512 actor + F9 window beam search residuals + H13b time-advance/window beam vs target_final_v12b_g15 on the wrist/finger phase weights; teacher only, not shipped" \
      --lineage-file $N8/code/kcem_n8.py --lineage-file $N8/code/port_n8.py --lineage-file $N8/code/build_world_n8.py $FL ;;
  30)
    FL="--lineage-file $F9/code/mresF9.py --lineage-file $F9/code/mlibF9.py --lineage-file $F9/code/wbeamF9.py --lineage-file $C22/streamD9_tight/code/wbeam9.py --lineage-file $C22/streamU8_15_16/code/mres.py $HL"
    export WORLD=$C22/streamE_ep25_30/ep30/world_T1
    $PY -B $F9/code/mkbase30.py $A $B "$NOTE"
    bash $N8/code/stage30.sbatch $B --lineage-file $N8/code/kcem_n8.py --lineage-file $F9/code/mkbase30.py --lineage-file $F9/code/f9_meshswap30.py $FL
    $PY -B $F9/code/f9_meshswap30.py $B/components/episode_000030 ;;
  10)
    source $C22/streamG_ep10/code/env.sh; W=$X/world_t15_d07; TEACHER=$X/kcem/B_LR
    $PY -B -u $SD/code/stage_G_component.py --episode 10 --distilled $A --base $B \
      --lineage-file $L/world_v1/recon_cup_thin_wall.json --lineage-file $L/code/build_recon_world.py --lineage-file $L/world_v1/source_assets/blue_cup.urdf --lineage-file $L/world_v1/source_assets/beige_cup.urdf \
      --lineage-file $SN/ep10/world_up/target_reference.json --lineage-file $SN/code/build_world_g10up.py --lineage-file $SD/code/glib.py --lineage-file $SN/code/port_n8.py --lineage-file $SN/code/kcem_n8.py \
      --lineage-file $W/collision_x8.json --lineage-file $W/source_assets/blue_cup.urdf --lineage-file $W/source_assets/beige_cup.urdf --lineage-file $X/code/thinlib.py --lineage-file $X/code/build_world_x8.py --lineage-file $X/code/swap_scene_x8.py --lineage-file $X/code/replay_x8.py --lineage-file $X/code/nesttest.py --lineage-file $TEACHER/result.json $( [ -f $TEACHER/settings.json ] && echo --lineage-file $TEACHER/settings.json ) $HL ;;
esac
echo STAGE_DONE
