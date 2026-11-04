#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=spoon15-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/spoon15-stage-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
legacy=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
root="$base/diagnostics/spoon_transfer/episode_000015"
teacher="$root/retention_window32_44"
"$legacy/.venv/bin/python" -u "$base/package_code/stage_seven_component.py" --episode 15 --actor-report "$teacher/actor_grid/completed.json" --video "$teacher/actor_grid/actor_c1024_p0.003_v0.0001/evaluation/videos" --lineage-file "$base/code/spoon_transfer.py" --lineage-file "$base/code/spoon_actor_grid.py" --lineage-file "$root/collision_v2_donor28_park1/spec.json" --lineage-file "$root/retention_after30/settings.json" --lineage-file "$root/retention_fine30/settings.json" --lineage-file "$root/retention_finer30/settings.json" --lineage-file "$teacher/settings.json" --lineage-file "$teacher/result.json"
