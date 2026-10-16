#!/usr/bin/env bash
#SBATCH --job-name=uniretarget-all
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
set -euo pipefail
RUN_ROOT=${1:?Prepared batch root required}
PHASE=${2:?Phase required}
EPISODE=$(python3 - "$RUN_ROOT" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
from pathlib import Path
print(json.loads((Path(sys.argv[1])/'batch_manifest.json').read_text())['episodes'][int(sys.argv[2])])
PY
)
SEED=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["seed"])' "$RUN_ROOT/batch_manifest.json")
exec bash "$RUN_ROOT/run.sh" "$PHASE" "$SEED" "$EPISODE"
