#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=node100
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o /data1/hansol/retarget33/logs/%x-%j.log
#SBATCH -J R33-stills
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
/home/hansol/v2d/.venvs/v2d-host/bin/python -B - <<'PY'
import cv2, numpy as np
V = '/data1/hansol/retarget33/videos/R2/ep33_R2_vs_HGA_c9.mp4'
picks = [(20, '1.0 s  spoon resting on the board'), (44, '2.2 s  right hand grasps the spoon'), (120, '6.0 s  scooping, spoon in hand'),
         (187, '9.35 s  spoon flipped over and shaken (our 9.2-9.5 s fix)'), (232, '11.6 s  spoon bowl inside the cup'),
         (259, '12.95 s  spoon edge-on, back side to camera (our fix)'), (300, '15.0 s  spoon lifted out'), (340, '17.0 s  end')]
cap = cv2.VideoCapture(V); fr = {}; i = 0; want = {p for p, _ in picks}
while True:
    ok, im = cap.read()
    if not ok: break
    if i in want: fr[i] = im
    i += 1
rows = []
for p, lab in picks:
    im = fr[p]; w = im.shape[1]; drop = w // 4 * 2; im = np.hstack([im[:, :w // 4], im[:, w // 4: w // 2], im[:, 3 * w // 4:]])  # original | target | R2 (drop HGA_c9 panel)
    bar = np.full((34, im.shape[1], 3), 24, np.uint8); cv2.putText(bar, lab, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 210, 255), 2, cv2.LINE_AA)
    rows.append(np.vstack([bar, im]))
g = np.vstack(rows); g = cv2.resize(g, (g.shape[1] * 2 // 3, g.shape[0] * 2 // 3), interpolation=cv2.INTER_AREA)
cv2.imwrite('/data1/hansol/retarget33/videos/R2/ep33_R2_keyframes.jpg', g, [cv2.IMWRITE_JPEG_QUALITY, 85]); print('OK', g.shape, i)
PY
