"""GT-free: temporal offset between ego_cam_a (RGB) and ego_cam_b / ego_cam_c (stereo pair) from global image-motion signals.
Usage camsync.py SPLIT EP. Positive lag = cam_b/c events happen later in frame index than in cam_a."""
import sys, cv2, numpy as np, json
split, ep = sys.argv[1], int(sys.argv[2])
base = f'/data1/video_to_data_challenge/track_3/{split}/videos/chunk-000'
def motion(cam):
    cap = cv2.VideoCapture(f'{base}/observation.images.{cam}/episode_{ep:06d}.mp4'); prev = None; sig = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(cv2.resize(fr, (320, int(320 * fr.shape[0] / fr.shape[1]))), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            fl = cv2.calcOpticalFlowFarneback(prev, g, None, .5, 3, 15, 3, 5, 1.2, 0)
            sig.append([np.median(fl[..., 0]), np.median(fl[..., 1]), np.median(np.linalg.norm(fl, axis=2))])
        else: sig.append([0, 0, 0])
        prev = g
    return np.array(sig)
S = {c: motion(c) for c in ['ego_cam_a', 'ego_cam_b', 'ego_cam_c']}
def xcorr(a, b, maxlag=4, sub=0.05):
    n = min(len(a), len(b)); a = a[:n]; b = b[:n]; t = np.arange(n); best = []
    for L in np.arange(-maxlag, maxlag + 1e-9, sub):
        bb = np.interp(t + L, t, b)  # b shifted: bb[t]=b[t+L]
        m = slice(maxlag + 1, n - maxlag - 1)
        aa = a[m] - a[m].mean(); cc = bb[m] - bb[m].mean()
        best.append((L, float((aa * cc).sum() / np.sqrt((aa ** 2).sum() * (cc ** 2).sum() + 1e-12))))
    best = np.array(best); i = best[:, 1].argmax(); return float(best[i, 0]), float(best[i, 1])
res = dict(ep=ep, split=split, frames={c: len(v) for c, v in S.items()})
for c in ['ego_cam_b', 'ego_cam_c']:
    res[c] = {k: xcorr(S['ego_cam_a'][:, j], S[c][:, j]) for j, k in enumerate(['dx', 'dy', 'mag'])}
for c in ['ego_cam_b','ego_cam_c']:
    a_=S['ego_cam_a']; b_=S[c]; n=min(len(a_),len(b_)); wins=[]
    for st in range(0, max(1,n-80), 40):
        sl=slice(st,min(n,st+80))
        wins.append([st]+[xcorr(a_[sl,j],b_[sl,j])[0] for j in range(2)])
    res[c+'_win']=wins
print(json.dumps(res))
json.dump(res, open(f'../out/camsyncw_{split}_{ep}.json', 'w'))
