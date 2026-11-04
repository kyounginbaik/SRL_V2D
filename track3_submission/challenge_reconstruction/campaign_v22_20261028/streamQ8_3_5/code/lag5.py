import sys, numpy as np, json
from scipy.ndimage import gaussian_filter1d as gf
z = np.load(sys.argv[1]); a = z['ach'][:, 1, :3] * 100; r = z['ref'][:, 1, :3] * 100; N = len(a)
w0, w1 = int(2.0 * 20), int(11.0 * 20)
print('window 2-11 s mean', np.linalg.norm(a - r, axis=1)[w0:w1].mean().round(3), 'outside', np.r_[np.linalg.norm(a - r, axis=1)[:w0], np.linalg.norm(a - r, axis=1)[w1:]].mean().round(3))
for lag in range(-6, 7):
    k = np.arange(w0, w1); print('lag', lag, 'frames: mean |a(t+lag)-r(t)|', np.linalg.norm(a[k + lag] - r[k], axis=1).mean().round(3))
# gain of the high-freq (stirring) content: a_hp = g * r_hp
for s in (4, 8):
    ah = (a - gf(a, s, axis=0, mode='nearest'))[w0:w1]; rh = (r - gf(r, s, axis=0, mode='nearest'))[w0:w1]
    g = (ah * rh).sum() / (rh * rh).sum(); print('hp sigma', s, 'gain', round(g, 3), 'rms ref', np.sqrt((rh ** 2).sum(1).mean()).round(3), 'rms ach', np.sqrt((ah ** 2).sum(1).mean()).round(3))
# spectrum of target velocity in window
v = np.diff(r[w0:w1], axis=0); F = np.abs(np.fft.rfft(v - v.mean(0), axis=0)) ** 2; f = np.fft.rfftfreq(len(v), 0.05)
P = F.sum(1); idx = np.argsort(P)[::-1][:6]; print('target vel spectrum peaks Hz', f[idx].round(2), (P[idx] / P.sum()).round(3))
ve = np.diff((a - r)[w0:w1], axis=0); Fe = (np.abs(np.fft.rfft(a[w0:w1] - r[w0:w1] - (a - r)[w0:w1].mean(0), axis=0)) ** 2).sum(1)
cum = np.cumsum(Fe) / Fe.sum(); print('error power cumulative by Hz:', {float(f[i].round(2)): float(cum[i].round(2)) for i in range(0, len(f), 5)})
