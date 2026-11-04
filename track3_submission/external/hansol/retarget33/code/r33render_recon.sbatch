#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=node100
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH -o /data1/hansol/retarget33/logs/render-%j.log
# (1) old recon (HGA7b, R2's target) render with R2 | R8; (2) latest recon (r6) render with R7 | R8; (3) Original | old recon | latest recon video
set -e
R=/data1/hansol/retarget33; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python; export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
RT=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/render_rollouts_titled.py
$PY -B - <<'PYX'
import json, numpy as np
from pathlib import Path
R = Path('/data1/hansol/retarget33'); K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33')
sc = json.load(open(K / 'runs/save_K33_cwHGAfix_ep33_HGA_c9/actor/evaluation/result.json'))['execution']['scene']
for od, world, a, b, ttl, vn in [
    ('recon_old', 'R33_HGA7b', 'R2', 'R8', ['Old recon (HGA7b): target of R2', 'R2: uploaded, MP-SR fail', 'R8 (shown on old recon)'], 'ep33_oldrecon.mp4'),
    ('R7R8', 'R33_HGA_r6', 'R7', 'R8', ['Latest recon (start roll -16): target of R5-R8', 'R7: + spoon-origin-in-cup cost', 'R8: + spoon-bowl-in-cup cost'], 'ep33_R7_vs_R8.mp4')]:
    O = R / 'videos' / od
    for n in (a, b):
        q = np.load(R / f'runs/eval_{n}/eval.npz')['Q']; np.savez(O / f'{n}_trajectory.npz', qpos=q, time_s=np.arange(len(q)) / 20.0)
    args = dict(episode_dir=str(R / f'worlds/{world}/ep33/episode_corrected/episode_000033'),
                baseline_report=dict(trajectory=str(O / f'{a}_trajectory.npz'), scene=sc), refined_report=dict(trajectory=str(O / f'{b}_trajectory.npz'), scene=sc),
                out_dir=str(O), width=480, height=360, stride=1, reference_scene=sc, titles=ttl, video_name=vn)
    (O / 'render_args.json').write_text(json.dumps(args, indent=1))
print('ARGS_OK')
PYX
for od in recon_old R7R8; do
  O=$R/videos/$od; rm -f $O/render_report.json; $PY -B $RT --args $O/render_args.json > $O/render.log 2>&1 & P=$!
  for i in $(seq 1 120); do [ -f $O/render_report.json ] && break; kill -0 $P 2>/dev/null || break; sleep 5; done; sleep 3; kill $P 2>/dev/null || true; ls -la $O/*.mp4
done
$PY -B - <<'PYX'
import cv2, numpy as np
from pathlib import Path
V = Path('/data1/hansol/retarget33/videos'); a = cv2.VideoCapture(str(V / 'recon_old/ep33_oldrecon.mp4')); b = cv2.VideoCapture(str(V / 'R7R8/ep33_R7_vs_R8.mp4'))
fps = a.get(cv2.CAP_PROP_FPS) or 20; w = None; n = 0
while True:
    ok1, f1 = a.read(); ok2, f2 = b.read()
    if not (ok1 and ok2): break
    P = f1.shape[1] // 4; fr = np.hstack([f1[:, :2 * P], f2[:, P:2 * P]])
    if w is None: w = cv2.VideoWriter(str(V / 'recon_cmp/ep33_oldrecon_vs_latest.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (fr.shape[1], fr.shape[0]))
    w.write(fr); n += 1
    if n in (1, 116, 232, 300): cv2.imwrite(str(V / f'recon_cmp/frame_{n:03d}.jpg'), fr)
w.release(); print('COMPOSED', n, 'frames')
PYX
echo RENDER_DONE
