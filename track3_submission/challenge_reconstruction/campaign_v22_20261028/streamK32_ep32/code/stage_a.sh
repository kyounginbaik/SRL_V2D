#!/bin/bash







# usage: [sbatch] stage.sbatch EP ACTOR_DIR BASE "note"
# The unchanged stagers (and teachers) that staged each v27 component (F9: N8 stage_n8 ep20, J8 stage_d ep28/38; C9: stageK 35/36, streamB stage_b 33,
# streamA stage_compose_component 5; D9: streamF_margins stage_compose_component ep6; K8 stageK ep37; H12 stage_b ep32) + H13a lineage. Writes only under BASE.
set -euo pipefail; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; H=$C22/streamH13a; R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent TMPDIR=$C22/streamK32_ep32/tmp
N8=$C22/streamN8_1_8_10_14_20_30; J8=$C22/streamJ8_19_25_28_38; F9=$C22/streamF9_tight; C9=$C22/streamC9_tight; D9=$C22/streamD9_tight; L8=$C22/streamL8_3_5_6; Q8=$C22/streamQ8_3_5
E=$1; A=$2; B=$3; NOTE=${4:-"H13a: v27 actor with a global phase time-advance and window-beam phase residuals, scored vs target_final_v12b_g15"}
HL="--lineage-file $H/code/h13lib.py --lineage-file $H/code/h13adv.py --lineage-file $H/code/h13res.py --lineage-file $H/code/h13scan.py --lineage-file $H/code/wbeamH13.py --lineage-file $H/code/c9beamH13.py --lineage-file $C22/streamK32_ep32/code/h13lib.py --lineage-file $C22/streamK32_ep32/code/mlibH.py --lineage-file $C22/streamK32_ep32/code/h13res.py --lineage-file $C22/streamK32_ep32/code/mresH.py --lineage-file $C22/streamK32_ep32/code/mres32H13.py --lineage-file $C22/streamK32_ep32/code/wbeamH.py --lineage-file $C22/streamK32_ep32/code/wbeamH13.py --lineage-file $C22/streamK32_ep32/code/c9beamH13.py --lineage-file $C22/streamK32_ep32/code/wbeam4H13.py --lineage-file $C22/streamK32_ep32/code/wbeamH14.py"
FL="--lineage-file $F9/code/mresF9.py --lineage-file $F9/code/mlibF9.py --lineage-file $F9/code/wbeamF9.py --lineage-file $D9/code/wbeam9.py --lineage-file $C22/streamU8_15_16/code/mres.py $HL"
C9L="--lineage-file $C9/code/c9res.py --lineage-file $C9/code/c9lib.py --lineage-file $C9/code/c9beam.py $HL"
cd $TMPDIR
case $E in
  20) $PY -B -u $N8/code/stage_n8.py --episode 20 --distilled $A --base $B --teacher $N8/ep20/kcem/k3_lr_init \
        --teacher-note "open-loop servo reference: stream D ep20 teacher re-planned (knot CEM kcem_n8.py) by stream N8; actor = F9_ep20_b4 actor + $NOTE; teacher only, not shipped" \
        --lineage-file $N8/code/kcem_n8.py $FL ;;
  28) source $J8/ep28/code/env.sh; export WORLD=world2
      $PY -B -u $SD/code/stage_d.py --episode 28 --distilled $A --base $B --teacher $J8/ep28/cem/w2_k3_sd8 \
        --teacher-note "open-loop servo reference: v21 (O6_ep28_C710) teacher ported into the target-referenced world2 and knot-CEM re-planned by J8; actor = F9_ep28_b1 actor + $NOTE; teacher only, not shipped" \
        --lineage-file $SD/code/kcem.py --lineage-file $SD/code/port_d.py --lineage-file $SD/code/build_world_d.py $FL ;;
  38) source $J8/ep38/code/env.sh; export WORLD=world0
      $PY -B -u $SD/code/stage_d.py --episode 38 --distilled $A --base $B --teacher $J8/ep38/ilc/g36/iter3 \
        --teacher-note "open-loop servo reference: v21 teacher ported into the target-referenced world0 and ILC re-planned by J8; actor = F9_ep38_b3 actor + $NOTE; teacher only, not shipped" \
        --lineage-file $SD/code/port_d.py --lineage-file $SD/code/build_world_d.py $FL ;;
  35|36)
    K=$C22/streamK8_35_36_37
    [ $E = 35 ] && T=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/teachers/ilc4/C1_none_gp0.3/iter0 || T=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep36/release3/seat17.0_rel18.5_lift19.1
    $PY -B -u $K/code/stageK.py --episode $E --distilled $A --base $B --teacher $T \
      --lineage-file $K/code/planK.py --lineage-file $K/code/cemK.py --lineage-file $K/code/ilcK.py --lineage-file $K/code/tlibK.py $C9L ;;
  37)
    K=$C22/streamK8_35_36_37
    $PY -B -u $K/code/stageK.py --episode 37 --distilled $A --base $B --teacher $R/refinement_20261018/expanded_batch/pickup_joint_margin/episode_000037/right_4 \
      --lineage-file $K/code/planK.py --lineage-file $K/code/cemK.py --lineage-file $K/code/ilcK.py --lineage-file $K/code/tlibK.py $HL ;;
  33)
    S=$C22/streamB_ep3233; O=/home/exaFLOPs/v22_overflow_20261028; T=$S/ep33/ilc2/gr0.3_gp0.4_w10.5_all/iter1
    $PY -B -u $C22/streamK32_ep32/code/stage_b_e31.py --episode 33 --distilled $A --base $B --teacher $T \
      --teacher-note "open-loop servo reference of B_ep33_c51 (streamB ilc2 iter1, not shipped); shipped actor = C9_ep33_beam4 actor (M8_ep33_r5 lineage + C9 window-beam phase residuals) + $NOTE" \
      --lineage-file $S/code/build_world_b.py --lineage-file $S/code/blib.py --lineage-file $C22/streamK32_ep32/code/mkref.py --lineage-file $C22/streamK32_ep32/code/stage_b_e31.py --lineage-file $S/code/fcem.py --lineage-file $T/result.json \
      --lineage-file $O/streamH_ep32/code/actor_res.py --lineage-file $O/streamI_ep33/code/release_cem.py --lineage-file $O/streamI_ep33/code/multi_cem.py \
      --lineage-file $C22/streamM8_32_33_15_16/code/mres.py --lineage-file $C22/streamM8_32_33_15_16/code/mlib.py $C9L ;;
  5)
    $PY -B -u $C22/streamA_ep35/code/stage_compose_component.py --episode 5 --distilled $A --base $B --teacher $C22/streamA_ep35/ep5/port/O_s4_st1.0_a0.5 \
      --lineage-file $L8/code/l8res.py --lineage-file $L8/code/l8rel.py --lineage-file $L8/code/plib.py --lineage-file $Q8/code/boost.py --lineage-file $Q8/code/wlot.py --lineage-file $Q8/code/q8ilc.py $C9L ;;
  6)
    $PY -B -u $C22/streamF_margins/ep6/code/stage_compose_component.py --episode 6 --distilled $A --base $B --teacher $C22/streamF_margins/ep6/port/O_s4_st0.5_a0.5 \
      --lineage-file $L8/code/l8res.py --lineage-file $L8/code/l8rel.py --lineage-file $L8/code/plib.py --lineage-file $Q8/code/boost.py --lineage-file $Q8/code/wlot.py --lineage-file $Q8/code/q8ilc.py --lineage-file $Q8/code/offgrid.py \
      --lineage-file $D9/code/mres9.py --lineage-file $D9/code/mlib9.py --lineage-file $D9/code/wbeam9.py --lineage-file $C22/streamU8_15_16/code/mres.py $HL ;;
  32)
    S=$C22/streamH12_ep32; O=/home/exaFLOPs/v22_overflow_20261028; V=$C22; T=$V/streamB_ep3233/ep32/fcem1/lk2.9_s3
    $PY -B -u $C22/streamK32_ep32/code/stage_b32_k32.py --episode 32 --distilled $A --base $B --teacher $T --teacher-note "H12_ep32_e1f0 actor + $NOTE" \
      --lineage-file $S/code32/build_world_b.py --lineage-file $S/code32/blib.py --lineage-file $V/streamB_ep3233/code/fcem.py --lineage-file $T/result.json \
      --lineage-file $O/streamH_ep32/code/actor_res.py --lineage-file $S/code32/mres.py --lineage-file $S/code32/mlib.py --lineage-file $S/code32/port.py \
      --lineage-file $S/code32/wbeam3.py --lineage-file $S/code32/wbeam4.py --lineage-file $C22/streamB9_ep32/code/wbeam3.py --lineage-file $S/code32/cmpworld.py --lineage-file $V/streamV8_ep32/code/wbeam2.py \
      --lineage-file $H/code/mres32H13.py --lineage-file $H/code/wbeam4H13.py $HL ;;
esac
echo STAGE_DONE
