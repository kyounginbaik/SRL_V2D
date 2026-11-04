"""GT-free camera-path selector cue: an object resting on the table must stay still in the world. For each camera path,
place the same object-in-camera track in the world and measure the object's world jitter over frames where it is at rest.
Rest frames are found without GT: the object's world speed is below a threshold under BOTH camera paths (i.e. the object
is not being carried), so the comparison only measures how much camera-motion leaks into a static object."""
import numpy as np
def rest_mask(Tw_a, Tw_b, fps=20., thr_cm_s=6.0, win=5):
    va = np.r_[0, np.linalg.norm(np.diff(Tw_a[:, :3, 3], axis=0), axis=1)] * 100 * fps
    vb = np.r_[0, np.linalg.norm(np.diff(Tw_b[:, :3, 3], axis=0), axis=1)] * 100 * fps
    k = np.ones(win) / win; va = np.convolve(va, k, 'same'); vb = np.convolve(vb, k, 'same')
    return (va < thr_cm_s) & (vb < thr_cm_s)
def jitter(Tw, m):
    """mean distance of rest-frame world positions from their segment median (cm), segments = runs of rest frames"""
    idx = np.where(m)[0]
    if len(idx) < 10: return np.nan, 0
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1); d = []
    for r in runs:
        if len(r) < 10: continue
        P = Tw[r, :3, 3]; d.extend(np.linalg.norm(P - np.median(P, 0), axis=1))
    return (float(np.mean(d)) * 100 if d else np.nan), len(d)
