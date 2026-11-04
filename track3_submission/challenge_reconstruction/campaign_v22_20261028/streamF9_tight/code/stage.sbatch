#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=F9-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamF9_tight/logs/stage-%j.log
# usage: sbatch stage.sbatch EP ACTOR_DIR BASE "note"
# Unchanged stage scripts of the streams that built each episode world (N8 stage_n8.py eps 8/20, J8 stage_d.py eps 28 (world2) / 38 (world0),
# N8/stream E stage30 path for ep30) + F9 lineage. ep30: after staging, A9_ep30_sweeper's submitted meshes are copied byte-identically (f9_meshswap30.py).
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; F9=$C22/streamF9_tight; N8=$C22/streamN8_1_8_10_14_20_30; J8=$C22/streamJ8_19_25_28_38
PY=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
E=$1; A=$2; B=$3; NOTE=${4:-"F9 window beam search on the phase weights"}
FL="--lineage-file $F9/code/mresF9.py --lineage-file $F9/code/mlibF9.py --lineage-file $F9/code/wbeamF9.py --lineage-file $C22/streamD9_tight/code/wbeam9.py --lineage-file $C22/streamU8_15_16/code/mres.py"
cd /tmp
case $E in
  8)  $PY -B -u $N8/code/stage_n8.py --episode 8 --distilled $A --base $B --teacher $N8/ep8/kcem/k_lr_20_105 \
        --teacher-note "open-loop servo reference: v21C teacher ported into the target_final_v5-referenced world (port_n8.py) and re-planned (knot CEM kcem_n8.py) by stream N8; actor = N8_ep08_klr512 actor + F9 window beam search residuals on the wrist/finger phase weights; teacher only, not shipped" \
        --lineage-file $N8/code/kcem_n8.py --lineage-file $N8/code/port_n8.py --lineage-file $N8/code/build_world_n8.py $FL ;;
  20) $PY -B -u $N8/code/stage_n8.py --episode 20 --distilled $A --base $B --teacher $N8/ep20/kcem/k3_lr_init \
        --teacher-note "open-loop servo reference: stream D ep20 teacher re-planned (knot CEM kcem_n8.py) by stream N8; actor = N8_ep20_k3c1024 actor + F9 window beam search residuals on the wrist/finger phase weights; teacher only, not shipped" \
        --lineage-file $N8/code/kcem_n8.py $FL ;;
  28) source $J8/ep28/code/env.sh; export WORLD=world2
      $PY -B -u $SD/code/stage_d.py --episode 28 --distilled $A --base $B --teacher $J8/ep28/cem/w2_k3_sd8 \
        --teacher-note "open-loop servo reference: v21 (O6_ep28_C710) teacher ported into the target-referenced world2 and knot-CEM re-planned by J8; actor = J8_ep28_w2k3c2048 actor + F9 window beam search residuals on the wrist/finger phase weights; teacher only, not shipped" \
        --lineage-file $SD/code/kcem.py --lineage-file $SD/code/port_d.py --lineage-file $SD/code/build_world_d.py $FL ;;
  38) source $J8/ep38/code/env.sh; export WORLD=world0
      $PY -B -u $SD/code/stage_d.py --episode 38 --distilled $A --base $B --teacher $J8/ep38/ilc/g36/iter3 \
        --teacher-note "open-loop servo reference: v21 teacher ported into the target-referenced world0 and ILC re-planned by J8; actor = J8_ep38_ilc36c2048 actor + F9 window beam search residuals on the wrist/finger phase weights; teacher only, not shipped" \
        --lineage-file $SD/code/port_d.py --lineage-file $SD/code/build_world_d.py $FL ;;
  30) export WORLD=$C22/streamE_ep25_30/ep30/world_T1
      $PY -B $F9/code/mkbase30.py $A $B "$NOTE"
      bash $N8/code/stage30.sbatch $B --lineage-file $N8/code/kcem_n8.py --lineage-file $F9/code/mkbase30.py --lineage-file $F9/code/f9_meshswap30.py $FL
      $PY -B $F9/code/f9_meshswap30.py $B/components/episode_000030 ;;
esac
echo STAGE_DONE
