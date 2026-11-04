#!/bin/bash
#SBATCH --partition=debug
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=ep1-carry-render
#SBATCH --output=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261018/logs/ep1-carry-render-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4
task_root=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017
selected="${1:-$task_root/refinement_20261018/episode_000001/reconstruction_repair/patch_height094/learned_carry_precision6}"
"$task_root/.venv/bin/python" "$task_root/code/render_rollouts.py" --args "$selected/render_args.json"
"$task_root/.venv/bin/python" - "$selected" <<'PY'
from pathlib import Path
import sys,json,cv2
p=Path(sys.argv[1]);video=p/'videos/reference_baseline_refined.mp4'
cap=cv2.VideoCapture(str(video));n=0;shape=None
fps=cap.get(cv2.CAP_PROP_FPS)
while True:
    ok,frame=cap.read()
    if not ok:break
    shape=frame.shape;n+=1
    if n in [25,37,46,61,91,121,136,146,156,168]:
        cv2.imwrite(str(p/f'videos/review_{n-1:04d}.jpg'),frame)
cap.release()
assert n==168,(n,fps,shape)
assert fps==10 and shape[:2]==(446,2048),(fps,shape)
(p/'videos/validation.json').write_text(json.dumps(dict(video=str(video),decoded_frames=n,fps=fps,width=shape[1],height=shape[0],source_timeline_seconds=[0.,16.7],rendered_from_actual_recorded_physics=True),indent=2)+'\n')
print(json.dumps(dict(decoded_frames=n,fps=fps,shape=shape)))
PY
