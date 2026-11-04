#!/bin/bash
#SBATCH -p debug
#SBATCH --exclude=node200
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /data1/hansol/retarget33/logs/render-%j.log
set -e
R=/data1/hansol/retarget33; O=$R/videos/M2E; K=/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33; T=/data1/hansol/teacher33
PY=/home/hansol/v2d/.venvs/v2d-host/bin/python; export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
RT=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/code/render_rollouts_titled.py
FF=$(ls /data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg* | head -1)
$PY -B - <<'PYX'
import json, numpy as np
from pathlib import Path
R = Path('/data1/hansol/retarget33'); O = R / 'videos/M2E'; T = Path('/data1/hansol/teacher33'); K = Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33')
sc = json.load(open(K / 'runs/save_K33_cwHGAfix_ep33_HGA_c9/actor/evaluation/result.json'))['execution']['scene']
rep = {}
for n in ('M1_C_f0', 'M2_end_fz161'):
    ex = json.load(open(T / f'runs/save_{n}/actor/evaluation/result.json'))['execution']; z = np.load(ex['trajectory'], allow_pickle=True)
    q = z['qpos'] if 'qpos' in z.files else z['Q']; np.savez(O / f'{n}_trajectory.npz', qpos=q, time_s=np.arange(len(q)) / 20.0); rep[n] = dict(trajectory=str(O / f'{n}_trajectory.npz'), scene=ex['scene'])
args = dict(episode_dir=str(R / 'worlds/R33_HGAc18/ep33/episode_corrected/episode_000033'), baseline_report=rep['M1_C_f0'], refined_report=rep['M2_end_fz161'],
            out_dir=str(O), width=480, height=360, stride=1, reference_scene=sc,
            titles=['Target: c18 recon', 'M1_C (uploaded, MP-SR fail)', 'M2_end_fz161: + put-down re-search + right-hand hold 16.1 s'], video_name='raw.mp4')
(O / 'render_args.json').write_text(json.dumps(args, indent=1)); print('ARGS_OK')
PYX
rm -f $O/render_report.json; $PY -B $RT --args $O/render_args.json > $O/render.log 2>&1 & P=$!
for i in $(seq 1 150); do [ -f $O/render_report.json ] && break; kill -0 $P 2>/dev/null || break; sleep 5; done; sleep 3; kill $P 2>/dev/null || true
$FF -y -loglevel error -i $O/raw.mp4 -c:v libx264 -pix_fmt yuv420p -crf 24 -preset fast -movflags +faststart $O/ep33_M2end_vs_M1C.mp4 && echo FF_OK
$PY -B - <<'PYX'
import cv2, numpy as np
cap = cv2.VideoCapture('/data1/hansol/retarget33/videos/M2E/raw.mp4'); want = [int(t*20) for t in (13.7, 14.8, 15.1, 15.3, 15.6, 16.5, 17.2)]; got = {}; k = 0
while True:
    ok, im = cap.read()
    if not ok: break
    if k in want: got[k] = im
    k += 1
rows = []
for f in want:
    im = got[f]; h, w = im.shape[:2]; im = cv2.resize(im, (w // 2, h // 2)); cv2.putText(im, f'{f/20:.2f}s', (10, im.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2); rows.append(im)
cv2.imwrite('/data1/hansol/retarget33/videos/M2E/M2E_end_strip.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85]); print('STRIP_OK')
PYX
