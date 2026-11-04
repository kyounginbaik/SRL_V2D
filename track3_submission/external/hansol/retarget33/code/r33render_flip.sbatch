#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH -o /data1/hansol/retarget33/logs/render-%j.log
# Original | c18 target (flipped spoon, our lineage) | H5r16 target (FINAL's ep33, v14 spoon direction); kinematic target panels only
set -e
R=/data1/hansol/retarget33; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python; export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
RT=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/render_rollouts_titled.py
FF=$(ls /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg* | head -1)
$PY -B - <<'PYX'
import json, numpy as np
from pathlib import Path
R = Path('/data1/hansol/retarget33'); K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33')
sc = json.load(open(K / 'runs/save_K33_cwHGAfix_ep33_HGA_c9/actor/evaluation/result.json'))['execution']['scene']
q = np.load(R / 'runs/eval_R12/eval.npz')['Q']
for sub, world, ttl in [('a', str(R / 'worlds/R33_HGAc18/ep33/episode_corrected/episode_000033'), 'c18 target: spoon FLIPPED (our lineage, used by M1_C)'),
                        ('b', '/data1/hansol/teacher33/worlds/R33H5r16_W1/ep33/episode_corrected/episode_000033', 'H5r16 target: v14 spoon direction (FINAL ep33)')]:
    O = R / 'videos/flip' / sub; np.savez(O / 'X_trajectory.npz', qpos=q, time_s=np.arange(len(q)) / 20.0)
    args = dict(episode_dir=world, baseline_report=dict(trajectory=str(O / 'X_trajectory.npz'), scene=sc), refined_report=dict(trajectory=str(O / 'X_trajectory.npz'), scene=sc),
                out_dir=str(O), width=480, height=360, stride=1, reference_scene=sc, titles=[ttl, 'robot (ignore)', 'robot (ignore)'], video_name='v.mp4')
    (O / 'render_args.json').write_text(json.dumps(args, indent=1))
print('ARGS_OK')
PYX
for s in a b; do O=$R/videos/flip/$s; rm -f $O/render_report.json; $PY -B $RT --args $O/render_args.json > $O/render.log 2>&1 & P=$!
  for i in $(seq 1 150); do [ -f $O/render_report.json ] && break; kill -0 $P 2>/dev/null || break; sleep 5; done; sleep 3; kill $P 2>/dev/null || true; done
$PY -B - <<'PYX'
import cv2, numpy as np
from pathlib import Path
V = Path('/data1/hansol/retarget33/videos/flip'); a = cv2.VideoCapture(str(V / 'a/v.mp4')); b = cv2.VideoCapture(str(V / 'b/v.mp4'))
fps = a.get(cv2.CAP_PROP_FPS) or 20; w = None; n = 0; keep = {int(t * 20): None for t in (6.5, 7.5, 8.5, 9.35, 12.5, 13.0)}
while True:
    ok1, f1 = a.read(); ok2, f2 = b.read()
    if not (ok1 and ok2): break
    P = f1.shape[1] // 4; fr = np.hstack([f1[:, :2 * P], f2[:, P:2 * P]])
    if w is None: w = cv2.VideoWriter(str(V / 'raw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (fr.shape[1], fr.shape[0]))
    w.write(fr)
    if n in keep:
        s = cv2.resize(fr, (fr.shape[1] // 2, fr.shape[0] // 2)); cv2.putText(s, f'{n/20:.2f}s', (10, s.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2); keep[n] = s
    n += 1
w.release(); cv2.imwrite(str(V / 'flip_strip.jpg'), np.vstack([v for v in keep.values() if v is not None]), [cv2.IMWRITE_JPEG_QUALITY, 85]); print('COMPOSED', n)
PYX
$FF -y -loglevel error -i $R/videos/flip/raw.mp4 -c:v libx264 -pix_fmt yuv420p -crf 26 -preset fast -movflags +faststart $R/videos/flip/ep33_spoon_flip_c18_vs_H5r16.mp4 && echo FF_OK
