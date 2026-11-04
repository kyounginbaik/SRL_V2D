#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=sweeper30-grasp
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/sweeper30-grasp-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json,subprocess
b=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019')
python='/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python'
def run(item):
    anchor,park=item;tag=f'donor{anchor:g}_park{int(park)}_v1'
    log=b/f'logs/sweeper30_{tag}.log'
    args=[python,'-u',str(b/'code/sweeper_transfer.py'),'--anchor',str(anchor),'--tag',tag]
    if park:args.append('--park-left')
    with log.open('w') as f:r=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
    print(json.dumps(dict(tag=tag,returncode=r.returncode,log=str(log))),flush=True)
    return r.returncode
with ThreadPoolExecutor(max_workers=4) as pool: codes=list(pool.map(run,[(3.,False),(3.,True),(5.,False),(5.,True)]))
if all(codes):raise SystemExit(1)
PY
