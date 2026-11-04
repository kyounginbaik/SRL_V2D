#!/bin/bash
#SBATCH --job-name=A-B2A-gen
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/home/hansol/v2d/outputs/variants_20261030/A_work/logs/gen-%A_%a.log
# guarded smoothing (guard_clip14.py) of one clip per array task; LIST lines = "public|eval EPISODE_DIR", EXTRA = extra args
set -uo pipefail
W=/home/hansol/v2d/outputs/variants_20261030/A_work
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${LIST:?}")
[ -z "$LINE" ] && exit 0
$W/code/genv.sh $W/code/guard_clip14.py $LINE ${PARAMS:?} ${EXTRA:-} || echo "GENFAIL $LINE"
echo "GENDONE $LINE"
