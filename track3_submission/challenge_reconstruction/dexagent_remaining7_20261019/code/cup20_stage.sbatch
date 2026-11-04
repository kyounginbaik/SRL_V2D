#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --job-name=cup20-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/cup20-stage-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
legacy=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
"$legacy/.venv/bin/python" -u "/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/package_code/stage_seven_component.py" --episode 20 --actor-report "/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/episode20_hold/final_pitcher_release_75_neutral/trial1/actor_grid/completed.json"  --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/cup20_feedback.py --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/release_refined_grasp.py --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/spoon_actor_grid.py --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/episode20_hold_prior.py --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/pitcher_release_grid.py --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/code/search_pitcher_pickup.py --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/pitcher_pickup/release_grid_v2/hold/result.json --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/episode20_hold/axis_feedback_v1/gain1.0_release11.0/result.json --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/episode20_hold/pitcher_feedback_v1/gain1.0_release11.0/result.json --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/episode20_hold/final_cup_release/trial1/result.json --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/episode20_hold/final_pitcher_release_75_neutral/trial1/result.json --lineage-file /data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/diagnostics/episode20_hold/final_pitcher_release_75_neutral/trial1/actor_grid/attempts.json
