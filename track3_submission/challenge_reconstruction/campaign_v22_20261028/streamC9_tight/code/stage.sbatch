#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=C9-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamC9_tight/logs/stage-%j.log
# usage: sbatch stage.sbatch EP ACTOR_DIR BASE_DIR
# Unchanged stagers of the streams that staged the v23b components (writes only under BASE):
#   35/36: streamK8 stageK.py (teacher as K8_ep35_ilc2 / K8_ep36_ecem_ilc); 33: streamB stage_b.py (as M8_ep33_r5); 5: streamA stage_compose_component.py (as Q8_ep05_g1)
set -euo pipefail; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; C9=$C22/streamC9_tight; R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent TMPDIR=$C9/tmp
E=$1; A=$2; B=$3
C9L="--lineage-file $C9/code/c9res.py --lineage-file $C9/code/c9lib.py --lineage-file $C9/code/c9beam.py"
cd $TMPDIR
case $E in
  35|36)
    K=$C22/streamK8_35_36_37
    [ $E = 35 ] && T=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamK_ep35/teachers/ilc4/C1_none_gp0.3/iter0 || T=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamG_ep36_30/ep36/release3/seat17.0_rel18.5_lift19.1
    $PY -B -u $K/code/stageK.py --episode $E --distilled $A --base $B --teacher $T \
      --lineage-file $K/code/planK.py --lineage-file $K/code/cemK.py --lineage-file $K/code/ilcK.py --lineage-file $K/code/tlibK.py $C9L ;;
  33)
    S=$C22/streamB_ep3233; O=/home/exaFLOPs/v22_overflow_20261028; T=$S/ep33/ilc2/gr0.3_gp0.4_w10.5_all/iter1
    $PY -B -u $S/code/stage_b.py --episode 33 --distilled $A --base $B --teacher $T \
      --teacher-note "open-loop servo reference of B_ep33_c51 (streamB ilc2 iter1, not shipped); shipped actor = M8_ep33_r5 actor (I_ep33_lr4 c51 weights + registered MultiReleasePolicy + M8 wrist/finger phase residuals) + stream C9 window-beam phase residuals (c9beam.py)" \
      --lineage-file $S/code/build_world_b.py --lineage-file $S/code/blib.py --lineage-file $S/code/fcem.py --lineage-file $T/result.json \
      --lineage-file $O/streamH_ep32/code/actor_res.py --lineage-file $O/streamI_ep33/code/release_cem.py --lineage-file $O/streamI_ep33/code/multi_cem.py \
      --lineage-file $C22/streamM8_32_33_15_16/code/mres.py --lineage-file $C22/streamM8_32_33_15_16/code/mlib.py $C9L ;;
  5)
    L8=$C22/streamL8_3_5_6; Q8=$C22/streamQ8_3_5
    $PY -B -u $C22/streamA_ep35/code/stage_compose_component.py --episode 5 --distilled $A --base $B --teacher $C22/streamA_ep35/ep5/port/O_s4_st1.0_a0.5 \
      --lineage-file $L8/code/l8res.py --lineage-file $L8/code/l8rel.py --lineage-file $L8/code/plib.py --lineage-file $Q8/code/boost.py --lineage-file $Q8/code/wlot.py --lineage-file $Q8/code/q8ilc.py $C9L ;;
esac
