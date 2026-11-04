"""Write render args + a node100 sbatch for 4-pane comparison videos: original video | kinematic reference |
submitted rollout (pot 200 g) | corrected rollout. Usage: render_pot_videos.py --out DIR EP=VARIANT_EVAL_DIR ..."""
import argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('pairs', nargs='+'); ap.add_argument('--submit', action='store_true')
ap.add_argument('--width', type=int, default=640); ap.add_argument('--height', type=int, default=480); ap.add_argument('--stride', type=int, default=1)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); jobs = []
for pair in a.pairs:
    e, var = pair.split('='); e = int(e); var = Path(var).resolve()
    base = json.load(open(ROOT / f'refinement_20261018/shared_policy_handoff/episode_{e:06d}/portable/original/evaluation_result.json'))
    refined = json.load(open(var / 'result.json'))
    assert refined['episode'] == f'episode_{e:06d}'
    vdir = out / f'episode_{e:06d}'; vdir.mkdir(exist_ok=True)
    args = dict(episode_dir=base['input_dir'], baseline_report=str(Path(base['execution']['scene']).parent / 'execution_report.json'),
                refined_report=str(Path(refined['execution']['scene']).parent / 'execution_report.json'), out_dir=str(vdir),
                width=a.width, height=a.height, stride=a.stride, include_source=True,
                titles=['Reference (kinematic)', 'Submitted rollout (pot 200 g)', 'Corrected rollout (carry wrist fix)'],
                video_name=f'episode_{e:02d}_submitted_vs_corrected.mp4')
    for k in ('baseline_report', 'refined_report'): assert Path(args[k]).exists(), args[k]
    json.dump(args, open(vdir / 'render_args.json', 'w'), indent=1); jobs.append(str(vdir / 'render_args.json'))
json.dump(jobs, open(out / 'render_jobs.json', 'w'), indent=1)
sb = f"""#!/usr/bin/env bash
#SBATCH --job-name=pot-render
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --nodelist=node100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --array=0-{len(jobs)-1}%4
#SBATCH --output={out}/render-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
args=$("{ROOT}/.venv/bin/python" -c "import json,sys; print(json.load(open('{out}/render_jobs.json'))[int(sys.argv[1])])" "$SLURM_ARRAY_TASK_ID")
srun --cpu-bind=cores "{ROOT}/.venv/bin/python" "{ROOT}/code/render_rollouts_titled.py" --args "$args"
"""
(out / 'render.sbatch').write_text(sb); print(len(jobs), 'render jobs ->', out / 'render.sbatch')
if a.submit:
    import subprocess; print(subprocess.run(['sbatch', str(out / 'render.sbatch')], capture_output=True, text=True).stdout)
