#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --job-name=expanded-retention
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/retention-%A_%a.log
set -euo pipefail
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
readarray -t args < <(.venv/bin/python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
from pathlib import Path
c=json.loads(Path('refinement_20261018/expanded_batch/retention_cases.json').read_text())[int(sys.argv[1])]
for k in ['episode','source','start_s','stop_s','protect']:print(c[k])
PY
)
episode=$(printf 'episode_%06d' "${args[0]}")
.venv/bin/python code/refine_versioned_retention.py --source "${args[1]}" \
 --episode-dir "episodes/${episode}" --output "refinement_20261018/expanded_batch/retention/${episode}" \
 --start-s "${args[2]}" --stop-s "${args[3]}" --protect-object "${args[4]}" \
 --samples 96 --iterations 20 --workers 24 --seed 63
