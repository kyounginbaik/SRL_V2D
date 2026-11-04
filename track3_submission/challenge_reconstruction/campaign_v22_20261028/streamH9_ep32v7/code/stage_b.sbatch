#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=H9-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH9_ep32v7/logs/stage-%j.log
# usage: sbatch stage_b.sbatch ACTOR_DIR BASE_DIR "NOTE"   (H9 copy of B9 stage_b.py; teacher = streamB fcem1/lk2.9_s3 as V8)
set -euo pipefail; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH9_ep32v7/tmp
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; S=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamH9_ep32v7; O=/home/exaFLOPs/v22_overflow_20261028; V=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent
T=$V/streamB_ep3233/ep32/fcem1/lk2.9_s3
cd $TMPDIR; $R/.venv/bin/python -B -u $S/code/stage_b.py --episode 32 --distilled $1 --base $2 --teacher $T --teacher-note "$3" \
  --lineage-file $S/code/build_world_b.py --lineage-file $S/code/blib.py --lineage-file $V/streamB_ep3233/code/fcem.py --lineage-file $T/result.json \
  --lineage-file $O/streamH_ep32/code/actor_res.py --lineage-file $S/code/mres.py --lineage-file $S/code/mlib.py --lineage-file $S/code/port.py \
  --lineage-file $S/code/wbeam3.py --lineage-file $S/code/wbeam4.py --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB9_ep32/code/wbeam3.py --lineage-file $S/code/cmpworld.py --lineage-file $V/streamV8_ep32/code/wbeam2.py
