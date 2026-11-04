#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node01
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=E-stage15
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamE_ep15/logs/stage-%j.log
# usage: sbatch stage15.sbatch STAGING_BASE TEACHER_RESULT_JSON
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /tmp && /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/package_code/stage_seven_component.py --episode 15 --base $1 --actor-report $1/actor_report.json --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamE_ep15/code/prepare_pickup15.py --lineage-file $2 --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamC_rigid_ilc/tools/actor_grid_c.py --lineage-file /data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamE_ep15/code/pickup15.sbatch
