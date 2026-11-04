#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=ep36-stage13
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/logs/stage_v13-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd /data1/SRL_V2D
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/code/stage_ep36_component.py --episode 36 --distilled /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261022_margins/ep36_v12warm_r2 --base /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/staging_v13 --teacher /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/search/roll_1_12.0_14.5 --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261022_margins/ep36_v12warm_r2/settings.json --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261022_margins/ep36_v12warm_r2/progress.json --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261022_margins/code/search_finger_margins_gen.py
