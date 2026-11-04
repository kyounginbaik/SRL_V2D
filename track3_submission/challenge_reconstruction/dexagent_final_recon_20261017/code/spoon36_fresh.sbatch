#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=spoon36-fresh-node100
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon36-fresh-node100-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
task_actor="$task_root/refinement_20261018/episode_000036/learned_pinky_margins_full"
"$task_root/.venv/bin/python" "$task_root/code/evaluate_policy_component.py" \
 --episode 36 --episode-dir "$task_root/episodes/episode_000036" \
 --scene "$task_actor/evaluation/execution/scene_physics.xml" --checkpoint "$task_actor/policy.npz" \
 --output "$task_actor/fresh_node100"
"$task_root/.venv/bin/python" "$task_root/code/audit_full_joint_limits.py" --run "$task_actor/fresh_node100"
"$task_root/.venv/bin/python" "$task_root/code/audit_object_support.py" "$task_actor/fresh_node100"
"$task_root/.venv/bin/python" - "$task_actor" <<'PY'
from pathlib import Path
import json,sys,numpy as np
r=Path(sys.argv[1]);a=np.load(r/'evaluation/execution/trajectory.npz');b=np.load(r/'fresh_node100/execution/trajectory.npz');d={}
for key in ['substep_ctrl','substep_qpos','substep_qvel']:
 d[key]=dict(bitwise_equal=bool(np.array_equal(a[key],b[key])),maximum_absolute_difference=float(abs(a[key]-b[key]).max()))
(r/'fresh_node100/cross_host_comparison.json').write_text(json.dumps(dict(source='node200 search final evaluation',fresh='node100 ComponentPolicy evaluator',arrays=d),indent=2)+'\n')
print(json.dumps(d))
PY
