#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=H14-svst
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH14_ep32/logs/%x-%j.log
# usage: sbatch svst.sbatch NAME TABLE.npz
#  save (bake the window-beam TABLE into the H12_ep32_e1f0 actor's phase weights; official evaluate_policy_component) -> stage (H12 stage_b unchanged + H14 lineage)
#  -> build-v2 (common_v2_candidates/run_one.sbatch, TAG H14_ep32_NAME) -> exactness, archive, mesh cmp vs e1f0, score_any vs target_final_v12b_g15 (+ v13_g16 if present)
set -euo pipefail
C22=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028; H=$C22/streamH14_ep32; R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; PY=$R/.venv/bin/python
A=$C22/streamH12_ep32/runs/save_e1f0/actor; COMP=$C22/streamH12_ep32/staging/ep32_e1f0/components/episode_000032; M=$COMP/meshes/episode_000032
N=$1; TB=$2; TAG=H14_ep32_$N; NM=episode_000032
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent TMPDIR=$H/tmp
SV=$H/runs/save_$N; B=$H/staging/ep32_$N
[ -f $SV/actor/evaluation/result.json ] || (cd $TMPDIR; $PY -B -u $H/code/mres32H13.py save --ep 32 --actor $A --meshes $M --out $SV --table $TB 2>&1 | grep -v -E "Warn|warn\(")
S=$C22/streamH12_ep32; O=/home/exaFLOPs/v22_overflow_20261028; T=$C22/streamB_ep3233/ep32/fcem1/lk2.9_s3
[ -f $B/portable_archives/${NM}_portable.tar.gz ] || (cd $TMPDIR; $PY -B -u $S/code32/stage_b.py --episode 32 --distilled $SV/actor --base $B --teacher $T \
  --teacher-note "H12_ep32_e1f0 actor + H14 monotone full-horizon window-beam phase residuals (table $TB), scored vs target_final_v12b_g15" \
  --lineage-file $S/code32/build_world_b.py --lineage-file $S/code32/blib.py --lineage-file $C22/streamB_ep3233/code/fcem.py --lineage-file $T/result.json \
  --lineage-file $O/streamH_ep32/code/actor_res.py --lineage-file $S/code32/mres.py --lineage-file $S/code32/mlib.py --lineage-file $S/code32/port.py \
  --lineage-file $S/code32/wbeam3.py --lineage-file $S/code32/wbeam4.py --lineage-file $C22/streamB9_ep32/code/wbeam3.py --lineage-file $S/code32/cmpworld.py --lineage-file $C22/streamV8_ep32/code/wbeam2.py \
  --lineage-file $H/code/mres32H13.py --lineage-file $H/code/h13lib.py --lineage-file $H/code/h13adv.py --lineage-file $H/code/wbeamH14.py --lineage-file $TB)
bash /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/common_v2_candidates/run_one.sbatch 32 $B/components/$NM $TAG
bash $H/code/vs.sh $TAG
