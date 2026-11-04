#!/bin/bash
#SBATCH -p debug
#SBATCH --nodelist=node100
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH -o /data1/hansol/ep32_annot/grid_%j.log
export PYTHONPATH=/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/.venv/lib/python3.11/site-packages
PY="/home/hansol/v2d/.venvs/v2d-host/bin/python -B"; O=/data1/hansol/ep32_annot; ANN=/home/hansol/v2d/outputs/ep32_robot_20261031/annot_json/ep32_f40_209.json
$PY $O/ep32_annot_crops.py $O/crops 140-220 2 $ANN
$PY - <<'PY'
import json, cv2, numpy as np
from pathlib import Path
O = Path('/data1/hansol/ep32_annot'); rows = json.loads((O / 'crops/frames.json').read_text()); tiles = []
for r in rows:
    im = cv2.imread(str(O / 'crops' / f"f{r['frame']:03d}.jpg")); im = cv2.resize(im, (440, 440 * im.shape[0] // im.shape[1]))
    bar = np.full((34, 440, 3), 255, np.uint8); cv2.putText(bar, f"f{r['frame']}  {r['time_s']:.2f}s", (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    tiles.append(np.vstack([bar, im]))
h = max(t.shape[0] for t in tiles); tiles = [np.vstack([t, np.full((h - t.shape[0], 440, 3), 255, np.uint8)]) for t in tiles]
C = 5; tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % C)
g = np.vstack([np.hstack(tiles[i:i + C]) for i in range(0, len(tiles), C)])
head = np.full((60, g.shape[1], 3), 255, np.uint8); cv2.putText(head, 'ep32 spoon, RGB frames f140-f220 (every 2 frames, 20 fps)   original video only', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite(str(O / 'ep32_frames_f140_220.jpg'), np.vstack([head, g]), [cv2.IMWRITE_JPEG_QUALITY, 85]); print('grid', g.shape, len(rows))
PY
