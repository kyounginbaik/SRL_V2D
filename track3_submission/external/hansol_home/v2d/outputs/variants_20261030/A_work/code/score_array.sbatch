#!/bin/bash
#SBATCH --job-name=A-B2A-score
#SBATCH --partition=debug
#SBATCH --exclude=node200
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/hansol/v2d/outputs/variants_20261030/A_work/logs/score-%A_%a.log
# unchanged public scorer (code/score.sh: score_bundle.py --metrics) on one export folder per array task; LIST = folders
set -uo pipefail
W=/home/hansol/v2d/outputs/variants_20261030/A_work
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${LIST:?}")
[ -z "$LINE" ] && exit 0
$W/code/score.sh "$LINE" && echo "DONE $LINE"
