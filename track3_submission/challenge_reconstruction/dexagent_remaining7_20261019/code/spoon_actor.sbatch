#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --job-name=spoon-actor
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/slurm/spoon-actor-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2
base=/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019
legacy=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
name=$(printf 'episode_%06d' "$SPOON_EPISODE")
source="$base/diagnostics/spoon_transfer/$name/$SPOON_SOURCE"
"$legacy/.venv/bin/python" -c 'import json,sys;r=json.load(open(sys.argv[1])); assert r["metrics"]["reference_proxy_mp_sr"]==1. and r["execution"]["physics_verified"],"Teacher must pass full independent physical replay before actor training"' "$source/result.json"
"$legacy/.venv/bin/python" -u "$base/code/refine_and_train.py" --episode "$SPOON_EPISODE" --source-run "$source" --base "$base/physics_collision_v2" --output "$source/learned_actor" --seed 44 --workers 4
"$legacy/.venv/bin/python" - "$source" "$base/physics_collision_v2" "$name" <<'PY'
import pathlib,json,sys
src,base,name=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
completed=src/'learned_actor/completed.json'
if not completed.exists(): raise SystemExit('Learned actor did not pass; inspect actor_attempts.json')
r=json.loads(completed.read_text()); ev=json.loads((pathlib.Path(r['evaluation'])/'result.json').read_text())
p=src/'learned_actor/videos/render_args.json';p.parent.mkdir(parents=True,exist_ok=True)
args=dict(episode_dir=str(base/'episodes'/name),baseline_report=str(base/'runs/baseline_v1'/name/'baseline/execution/execution_report.json'),refined_report=ev['execution'],out_dir=str(p.parent),width=512,height=384,stride=1,include_source=True)
p.write_text(json.dumps(args,indent=2)+'\n')
PY
"$legacy/.venv/bin/python" -u "$legacy/code/render_rollouts.py" --args "$source/learned_actor/videos/render_args.json"
