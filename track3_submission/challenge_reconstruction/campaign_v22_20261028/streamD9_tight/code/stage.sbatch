#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=D9-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamD9_tight/logs/stage-%j.log
# usage: sbatch stage.sbatch EP ACTOR_DIR BASE "note"
# Unchanged stage scripts of the streams that built each episode world (same calls as Q8/L8 (eps 3/6), T8 (ep19), J8 (ep25)) + D9 lineage.
# ep25: after staging, the submitted board meshes are replaced by A9_ep25_boards' (byte-identical copy; d9_meshswap25.py).
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; D9=$C22/streamD9_tight; L8=$C22/streamL8_3_5_6; Q8=$C22/streamQ8_3_5
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code:/data1/SRL_V2D/third_party/DexAgent
E=$1; A=$2; B=$3; NOTE=${4:-"D9 window beam search on the phase weights"}
DL="--lineage-file $D9/code/mres9.py --lineage-file $D9/code/mlib9.py --lineage-file $D9/code/wbeam9.py --lineage-file $C22/streamU8_15_16/code/mres.py"
cd /tmp
case $E in
  3|6)
    if [ $E = 3 ]; then S=$C22/streamA_ep35/code/stage_compose_component.py; TE=$C22/streamA_ep35/ep3/ret/z6s4_r2.5_s21
    else S=$C22/streamF_margins/ep6/code/stage_compose_component.py; TE=$C22/streamF_margins/ep6/port/O_s4_st0.5_a0.5; fi
    $PY -B -u $S --episode $E --distilled $A --base $B --teacher $TE \
      --lineage-file $L8/code/l8res.py --lineage-file $L8/code/l8rel.py --lineage-file $L8/code/plib.py --lineage-file $Q8/code/boost.py --lineage-file $Q8/code/wlot.py --lineage-file $Q8/code/q8ilc.py --lineage-file $Q8/code/offgrid.py $DL ;;
  19)
    source $C22/streamJ8_19_25_28_38/ep19/code/env.sh; W=$J/world_a40; T=$J/cem/a40_kA_LR_2_5.5
    $PY -B -u $J/code/stage_ep19J.py --episode 19 --episode-dir $W/episode_target/episode_000019 --distilled $A --base $B --teacher $T \
      --teacher-note "open-loop servo reference teacher: stream F F_ep19_twin3 teacher (C_ep19_ilc lineage, twin port) re-ported / knot-CEM refined by J8 in $W; actor = J8 a40m2e chain + T8 rigid-hold ILC / CEM residuals + D9 window beam search on the wrist phase weights; teacher only, not shipped" \
      --lineage-file $J/code/port19J.py --lineage-file $J/code/build_world19J.py --lineage-file $J/code/mk_alpha.py --lineage-file $J/code/topple.py --lineage-file $J/code/kcem19.py \
      --lineage-file $J/sweeper_alpha.npz --lineage-file $J/sweeper_P.npz --lineage-file $F/code/common19F.py --lineage-file $F/twin_poses.npz --lineage-file $W/scene/scene_change.json \
      --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py --lineage-file $J/code/actor_res19.py --lineage-file $J/code/merge_actor.py \
      --lineage-file $C22/streamT8_ep19/code/ilc19t.py --lineage-file $C22/streamT8_ep19/code/combo.py $DL ;;
  25)
    Q=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamQ_ep25; EE=$C22/streamE_ep25_30; W=$EE/ep25/world_T1; J=$C22/streamJ8_19_25_28_38/ep25
    $PY -B $D9/code/mkbase25.py $A $B "$NOTE"
    $PY -u $Q/code/stage_seven_component_corrected_boards.py --episode 25 --base $B --actor-report $B/actor_report.json --lineage-file $W/lineage.json --lineage-file $W/source_corrected/episode_000025/board_track_correction.json --lineage-file $EE/code25/build_worldE.py --lineage-file $EE/code/conv25.py --lineage-file $EE/code/elib.py --lineage-file $Q/code/make_slotted_mesh_C.py --lineage-file $Q/code/cut_slot_body.py --lineage-file $Q/meshes_slotted_C/slotted_mesh_report.json --lineage-file $J/code/res25.py $DL
    $PY -B $D9/code/d9_meshswap25.py $B/components/episode_000025 ;;
esac
echo STAGE_DONE
