#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --array=0-5%6
#SBATCH --job-name=episode19-handle-pickup
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/handle19-%A_%a.log
set -euo pipefail
cd /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
readarray -t fields < <(.venv/bin/python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
from pathlib import Path
c=json.loads(Path('refinement_20261018/expanded_batch/handle_pickup_cases.json').read_text())[int(sys.argv[1])]
print(c['spec']);print(c['output'])
PY
)
.venv/bin/python code/physics_pickup_search_v2.py --spec "${fields[0]}" --output "${fields[1]}" \
 --samples 64 --iterations 16 --workers 12 --seed 44
