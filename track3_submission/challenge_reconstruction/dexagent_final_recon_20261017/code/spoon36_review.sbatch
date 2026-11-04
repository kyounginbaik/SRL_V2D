#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=spoon36-video-review
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/spoon36-video-review-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
export LD_LIBRARY_PATH="$task_root/.software_gl/usr/lib64:${LD_LIBRARY_PATH:-}"
task_run="$1"
"$task_root/.venv/bin/python" "$task_root/code/audit_episode3_grip_quality.py" --run "$task_run" --sample-stride 5
"$task_root/.venv/bin/python" - "$task_root" "$task_run" <<'PY'
import json,sys
from pathlib import Path
b,r=map(Path,sys.argv[1:]);args=dict(episode_dir=str(b/'episodes/episode_000036'),baseline_report=str(b/'refinement_20261018/spoon36_long_grasp/search/execution/execution_report.json'),refined_report=str(r/'execution/execution_report.json'),out_dir=str(r/'videos'),width=480,height=360,stride=2,include_source=True)
(r/'render_arguments.json').write_text(json.dumps(args,indent=2)+'\n')
PY
"$task_root/.venv/bin/python" "$task_root/code/render_rollouts.py" --args "$task_run/render_arguments.json"
"$task_root/.venv/bin/python" - "$task_run" <<'PY'
import sys,cv2
from pathlib import Path
r=Path(sys.argv[1]);cap=cv2.VideoCapture(str(r/'videos/reference_baseline_refined.mp4'))
for time in [2.,5.,9.5,10.5,13.,15.,17.,18.5,19.9]:
 cap.set(cv2.CAP_PROP_POS_MSEC,time*1000);ok,image=cap.read()
 if not ok:raise RuntimeError(time)
 cv2.imwrite(str(r/f'videos/review_{time:04.1f}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,94])
cap.release()
PY
