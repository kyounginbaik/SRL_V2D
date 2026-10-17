#!/bin/bash
#SBATCH --partition=debug
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --array=0-11%8
#SBATCH --job-name=expanded-spoon-pickup
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/expanded_batch/logs/pickup-%A_%a.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
cd "$task_root"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
readarray -t case_fields < <(.venv/bin/python - "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
from pathlib import Path
case=json.loads(Path('refinement_20261018/expanded_batch/pickup_cases.json').read_text())[int(sys.argv[1])]
print(case['spec']);print(case['output'])
PY
)
.venv/bin/python code/physics_pickup_search_v2.py --spec "${case_fields[0]}" --output "${case_fields[1]}" --samples 64 --iterations 16 --workers 16 --seed 44
