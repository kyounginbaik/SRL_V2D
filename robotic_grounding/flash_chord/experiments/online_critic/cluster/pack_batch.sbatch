#!/usr/bin/env bash
#SBATCH --job-name=uniretarget-pack
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
export CONTAINER_MEMORY=6g
exec bash "${1:?Prepared batch root required}/run.sh" "pack-${2:?Phase required}"
