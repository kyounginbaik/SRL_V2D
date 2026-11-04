#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=B3233-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamB_ep3233/logs/stage-%j.log
# usage: sbatch stage.sbatch EP ACTOR_DIR BASE_DIR TEACHER_DIR "NOTE"
set -euo pipefail; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017; S=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamT32_teacher
export PYTHONPATH=$R/code:/data1/SRL_V2D/third_party/DexAgent
cd /tmp; $R/.venv/bin/python -B -u $S/code/stage_b.py --episode $1 --distilled $2 --base $3 --teacher $4 --teacher-note "$5" \
  --lineage-file $S/code/build_world_b.py --lineage-file $S/code/blib.py --lineage-file $S/code/delta_port.py --lineage-file $S/code/prepare_pickup_b.py --lineage-file $S/code/teacher_ilc_b.py --lineage-file $4/result.json
