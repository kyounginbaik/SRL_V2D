#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=spoon16-stage
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/spoon16-stage-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
legacy=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
teacher="$base/diagnostics/spoon_transfer/episode_000016/retention_after38"
"$legacy/.venv/bin/python" -u "$base/package_code/stage_seven_component.py" --episode 16 --actor-report "$teacher/actor_grid/completed.json" --video "$teacher/actor_grid/actor_c2048_p0.001_v5e-05/evaluation/videos" --lineage-file "$base/code/spoon_transfer.py" --lineage-file "$base/code/spoon_actor_grid.py" --lineage-file "$base/diagnostics/spoon_transfer/episode_000016/collision_v2_donor28_park1/spec.json" --lineage-file "$teacher/settings.json" --lineage-file "$teacher/result.json"
