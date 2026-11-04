#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=pot28-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/pot28-stage-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
legacy=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
root="$base/diagnostics/pot_transfer"
teacher="$root/pinch_roll180"
"$legacy/.venv/bin/python" -u "$base/package_code/stage_seven_component.py" --episode 28 --actor-report "$teacher/actor_grid/completed.json" --video "$teacher/actor_grid/actor_c512_p0.003_v0.0001/evaluation/videos" --lineage-file "$base/code/pot_transfer.py" --lineage-file "$base/code/pot_rest_grasp.py" --lineage-file "$base/code/pot_pinch_search.py" --lineage-file "$base/code/spoon_actor_grid.py" --lineage-file "$root/left_donor_park_right/mesh_registration.json" --lineage-file "$root/left_donor_park_right/spec.json" --lineage-file "$root/rest_roll180/settling.json" --lineage-file "$root/rest_roll180/grasp_roll.json" --lineage-file "$teacher/settings.json" --lineage-file "$teacher/result.json"
