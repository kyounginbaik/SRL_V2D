#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=cup14-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/cup14-stage-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
legacy=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
root="$base/diagnostics/cup_transfer"
teacher="$root/release_neutral/trial2"
"$legacy/.venv/bin/python" -u "$base/package_code/stage_seven_component.py" --episode 14 --actor-report "$teacher/actor_grid/completed.json" --video "$teacher/actor_grid/actor_c1024_p0.001_v5e-05/evaluation/videos" --lineage-file "$base/code/cup_transfer.py" --lineage-file "$base/code/release_refined_grasp.py" --lineage-file "$base/code/spoon_actor_grid.py" --lineage-file "$root/rest03_radial_hold/spec.json" --lineage-file "$root/rest03_radial_hold/result.json" --lineage-file "$teacher/result.json" --lineage-file "$teacher/actor_grid/attempts.json"
