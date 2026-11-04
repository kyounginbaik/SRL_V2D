#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=spoon-render
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/spoon-render-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
legacy=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
name=$(printf 'episode_%06d' "$SPOON_EPISODE")
source="$base/diagnostics/spoon_transfer/$name/$SPOON_SOURCE"
"$legacy/.venv/bin/python" - "$source" "$base/physics_collision_v2" "$name" <<'PY'
import pathlib,json,sys
src,base,name=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
r=json.loads((src/'result.json').read_text())
p=src/'videos/render_args.json';p.parent.mkdir(parents=True,exist_ok=True)
args=dict(episode_dir=str(base/'episodes'/name),baseline_report=str(base/'runs/baseline_v1'/name/'baseline/execution/execution_report.json'),refined_report=r['execution'],out_dir=str(p.parent),width=512,height=384,stride=2,include_source=True)
p.write_text(json.dumps(args,indent=2)+'\n')
PY
"$legacy/.venv/bin/python" -u "$legacy/code/render_rollouts.py" --args "$source/videos/render_args.json"
