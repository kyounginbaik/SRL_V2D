import cv2, json, numpy as np, sys
D = '/data1/hansol/ep33_20261102/orient/crops'; rows = json.load(open(D + '/frames.json'))
def sheet(lo, hi, cols, out):
    fr = [r for r in rows if lo <= r['frame'] <= hi]; S = 440; L = 44
    n = len(fr); R = (n + cols - 1) // cols
    img = np.full((70 + R * (S + L), cols * S, 3), 255, np.uint8)
    cv2.putText(img, f'ep33 spoon, RGB frames f{lo}-f{hi} (every 2 frames, 20 fps)   original video only', (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2, cv2.LINE_AA)
    for k, r in enumerate(fr):
        y, x = 70 + (k // cols) * (S + L), (k % cols) * S
        c = cv2.imread(f"{D}/f{r['frame']:03d}.jpg"); c = cv2.resize(c, (S, int(S * c.shape[0] / c.shape[1])))[:S]
        img[y + L:y + L + c.shape[0], x:x + S] = c
        cv2.rectangle(img, (x, y), (x + S - 1, y + L + S - 1), (200, 200, 200), 2)
        cv2.putText(img, f"f{r['frame']}  {r['time_s']:.2f}s", (x + 10, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 90]); print(out, img.shape)
sheet(176, 224, 5, '/data1/hansol/ep33_20261102/orient/ep33_frames_f176_224.jpg')
sheet(240, 280, 6, '/data1/hansol/ep33_20261102/orient/ep33_frames_f240_280.jpg')
