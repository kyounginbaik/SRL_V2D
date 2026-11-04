#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --job-name=sweeper30-clear
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/sweeper30-clear-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,json
b=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019');python='/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python'
def run(item):
    digit,angle=item;log=b/f'logs/sweeper30_fold_{digit}_{angle:g}.log'
    with log.open('w') as f:r=subprocess.run([python,'-u',str(b/'code/fold_sweeper_digits.py'),'--digits',digit,'--angle',str(angle)],stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps(dict(digits=digit,angle=angle,returncode=r.returncode,log=str(log))),flush=True);return r.returncode
with ThreadPoolExecutor(max_workers=6) as pool:codes=list(pool.map(run,[('both',1.),('both',1.4),('both',1.8),('pinky',1.4),('pinky',1.8),('ring',1.4)]))
if any(codes):raise SystemExit(1)
PY
