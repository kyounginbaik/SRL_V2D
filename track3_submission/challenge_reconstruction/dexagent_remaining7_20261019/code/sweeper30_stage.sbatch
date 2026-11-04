#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --job-name=sweeper30-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/sweeper30-stage-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
legacy=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
root="$base/physics_collision_v2/sweeper_transfer/liftclear_0.06_s2.5_v1"
teacher="$root/release_neutral/trial1"
"$legacy/.venv/bin/python" -u "$base/package_code/stage_seven_component.py" --episode 30 --actor-report "$teacher/actor_grid/completed.json" --video "$base/physics_collision_v2/review/episode_000030/final_release_actor" --lineage-file "$base/code/sweeper_transfer.py" --lineage-file "$base/code/lift_sweeper_clearance.py" --lineage-file "$base/code/release_refined_grasp_sweeper_snapshot.py" --lineage-file "$base/code/spoon_actor_grid.py" --lineage-file "$root/result.json" --lineage-file "$teacher/result.json" --lineage-file "$teacher/actor_grid/attempts.json" --lineage-file "$base/review/episode30_final_actor_contacts.json"
