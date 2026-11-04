#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node100
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --job-name=sweeper30-policyvid
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/sweeper30-policyvid-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/bin/python -u - <<'PY'
from pathlib import Path
import json,subprocess,sys
b=Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019');work=b/'physics_collision_v2'
actor=work/'sweeper_transfer/liftclear_0.06_s2.5_v1/actor_grid/completed.json'
selection=json.loads(actor.read_text())
subprocess.run([sys.executable,str(b/'code/render_selected.py'),'--episode','30','--episode-dir',selection['episode_dir'],
    '--baseline',str(work/'runs/baseline_v1/episode_000030/baseline'),'--refined',selection['evaluation'],
    '--output',str(work/'review/episode_000030/selected_actor'),'--stride','1'],check=True)
PY
