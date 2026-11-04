"""Draw the WiLoR hand prediction (joints_2d, undistorted RGB pixels) on the undistorted RGB frames of a prepared episode.
Writes <work_dir>/wilor_viz.mp4 and wilor_sheet.jpg.  usage: viz_wilor.py --ep 1 --obj white_pot [--step 12]"""
import argparse, os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import work_dir
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
ap = argparse.ArgumentParser(); ap.add_argument('--ep', type=int, required=True); ap.add_argument('--obj', required=True); ap.add_argument('--step', type=int, default=12); ap.add_argument('--npz', default='wilor_hands.npz'); ap.add_argument('--suffix', default='')
a = ap.parse_args(); wd = work_dir(a.ep, a.obj)
z = np.load(wd / a.npz); fr = {int(f): i for i, f in enumerate(z['frames'])}
imgs = sorted((wd / 'rgb_undist').glob('*.jpg')); n = len(imgs)
im0 = cv2.imread(str(imgs[0])); H, W = im0.shape[:2]; sc = 960 / W
vw = cv2.VideoWriter(str(wd / f'wilor_viz{a.suffix}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (int(W * sc), int(H * sc))); tiles = []
for f in range(n):
    im = cv2.imread(str(imgs[f]))
    if f in fr:
        i = fr[f]; j = z['joints_2d'][i]; right = z['is_right'][i] > 0.5; col = (0, 200, 255) if right else (255, 120, 0)
        x0, y0, x1, y1 = z['bbox'][i].astype(int); cv2.rectangle(im, (x0, y0), (x1, y1), col, 2)
        for a_, b_ in BONES: cv2.line(im, tuple(np.round(j[a_]).astype(int)), tuple(np.round(j[b_]).astype(int)), col, 3)
        for p in j: cv2.circle(im, tuple(np.round(p).astype(int)), 5, (255, 255, 255), -1)
        cv2.putText(im, f'frame {f}  {"RIGHT" if right else "LEFT"} hand  obj-IoU {z["obj_iou"][i]:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 3)
    else:
        cv2.putText(im, f'frame {f}  no hand (dropped)', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
    im = cv2.resize(im, (int(W * sc), int(H * sc))); vw.write(im)
    if f % a.step == 0: tiles.append(cv2.resize(im, (480, int(480 * H / W))))
vw.release()
cols = 4; rows = (len(tiles) + cols - 1) // cols; blank = np.zeros_like(tiles[0])
sheet = np.vstack([np.hstack((tiles + [blank] * (rows * cols - len(tiles)))[r * cols:(r + 1) * cols]) for r in range(rows)])
cv2.imwrite(str(wd / f'wilor_sheet{a.suffix}.jpg'), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(f'{len(fr)}/{n} frames with a hand; right-hand fraction {np.mean(z["is_right"]):.2f}; wrote {wd / f"wilor_viz{a.suffix}.mp4"} and wilor_sheet{a.suffix}.jpg')
