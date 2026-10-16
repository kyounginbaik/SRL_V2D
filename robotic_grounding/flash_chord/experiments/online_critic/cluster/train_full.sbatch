#!/usr/bin/env bash
#SBATCH --job-name=uniretarget-online
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
set -euo pipefail
# Pass an absolute prepared run directory; sbatch executes a spool copy of this file.
RUN_ROOT=${1:?Usage: sbatch train_full.sbatch /absolute/run/root [seed]}
SEED=${2:-43}
exec bash "$RUN_ROOT/run.sh" full "$SEED"
