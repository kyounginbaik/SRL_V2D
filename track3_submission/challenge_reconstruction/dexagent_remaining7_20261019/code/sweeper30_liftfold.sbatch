#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=sweeper30-lift
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/sweeper30-lift-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess,json,itertools
b=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019');python='/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python'
def run(item):
    digit,angle,start=item;log=b/f'logs/sweeper30_liftfold_{digit}_{angle:g}_s{start:g}.log'
    with log.open('w') as f:r=subprocess.run([python,'-u',str(b/'code/fold_sweeper_lift.py'),'--digits',digit,'--angle',str(angle),'--start',str(start)],stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps(dict(digits=digit,angle=angle,start=start,returncode=r.returncode)),flush=True);return r.returncode
with ThreadPoolExecutor(max_workers=8) as pool:codes=list(pool.map(run,itertools.product(['both','pinky'],[1.6,1.8],[2.25,2.45])))
if any(codes):raise SystemExit(1)
PY
