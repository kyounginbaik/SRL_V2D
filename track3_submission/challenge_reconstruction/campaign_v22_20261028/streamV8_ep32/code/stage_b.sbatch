#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=V8-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamV8_ep32/logs/stage-%j.log
# ep32/33: usage: sbatch stage_b.sbatch EP ACTOR_DIR BASE_DIR TEACHER_DIR "NOTE"   (streamB stage_b.py unchanged)
set -euo pipefail; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamV8_ep32/tmp
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; S=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB_ep3233; O=/home/exaFLOPs/v22_overflow_20261028
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent
cd $TMPDIR; $R/.venv/bin/python -B -u $S/code/stage_b.py --episode $1 --distilled $2 --base $3 --teacher $4 --teacher-note "$5" \
  --lineage-file $S/code/build_world_b.py --lineage-file $S/code/blib.py --lineage-file $S/code/fcem.py --lineage-file $4/result.json \
  --lineage-file $O/streamH_ep32/code/actor_res.py --lineage-file $O/streamI_ep33/code/release_cem.py --lineage-file $O/streamI_ep33/code/multi_cem.py \
  --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamM8_32_33_15_16/code/mres.py --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamM8_32_33_15_16/code/mlib.py --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamV8_ep32/code/wbeam.py --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamV8_ep32/code/wbeam2.py
