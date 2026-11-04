#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=spoon-transfer-v2
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/spoon-%j.log
set -euo pipefail
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
task_python=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$task_root"
"$task_python" -u code/prepare_variant.py --episode "$SPOON_EPISODE" \
  --source-root "$task_root/collision_recon" --workspace "$task_root/physics_collision_v2" --baseline
"$task_python" -u - <<'PY'
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os, subprocess, json
root=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019')
python='/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python'
episode=int(os.environ['SPOON_EPISODE'])
episode_dir=root/f'physics_collision_v2/episodes/episode_{episode:06d}'
def run(case):
    anchor,park=case
    tag=f'collision_v2_donor{int(anchor*10)}_park{int(park)}'
    log=root/f'diagnostics/spoon_transfer/episode_{episode:06d}/{tag}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    args=[python,'-u',str(root/'code/spoon_transfer.py'),'--episode',str(episode),'--episode-dir',str(episode_dir),'--anchor',str(anchor),'--tag',tag]
    if park:args.append('--park-left')
    with log.open('w') as handle:
        result=subprocess.run(args,stdout=handle,stderr=subprocess.STDOUT)
    report={'episode':episode,'tag':tag,'exit_code':result.returncode,'log':str(log)}
    print(json.dumps(report),flush=True)
    return report
with ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(run,[(2.8,False),(2.8,True),(3.5,False),(3.5,True)]))
if any(result['exit_code'] for result in results):raise SystemExit(1)
PY
