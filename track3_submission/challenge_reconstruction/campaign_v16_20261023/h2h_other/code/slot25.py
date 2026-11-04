"""ep25: where is each track's board slot in the image, and does the video show the notch there?
ours slot = -z body half (cut3mm collision report 'slot_open -mid'; the visual mesh has no slot), new slot = +z body half (their
parametric mesh). Per frame: angle between the two tracks' slot-mouth directions in the camera, and for each track the fraction
of its projected slot strip (board mid-plane, 0.6 cm wide, from the centre to the edge) that the SAM2 board mask covers (a real
slot should be EMPTY when visible, i.e. low coverage), vs the same for the mirrored strip (solid wood -> high coverage).
usage: slot25.py STRIDE OUT.json [FIGDIR]"""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__)); from common import *
stride, out = int(sys.argv[1]), sys.argv[2]; figdir = sys.argv[3] if len(sys.argv) > 3 else None
o = load_ours(25); n = load_new(25, o['names']); B = CR / 'public_gt_check_20261020/boards/episode_000025'
ev = Evidence(25, o['names'], {'wooden_piece_1': B / 'A/masks', 'wooden_piece_2': B / 'B/masks'})
SLOT = {'ours': np.array([0, 0, -1.]), 'new': np.array([0, 0, 1.])}
def centre(mesh): return np.asarray(mesh.vertices).mean(0)
def strip(A, c, d, sign, half_len=0.058, depth=0.055):
    """points of the slot strip in the camera: from the centre towards sign*d over 'depth' (slot depth ~ half width)."""
    s = np.linspace(0.006, depth, 25); P = c[None] + (sign * d)[None] * s[:, None]
    return P @ A[:3, :3].T + A[:3, 3]
cap = cv2.VideoCapture(str(HAW / 'episode_000025/wooden_piece_1/anycalib/undistorted.mp4')); rows = []
for t in range(0, 356, stride):
    row = dict(frame=t); cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, img = cap.read()
    if ok and img.shape[1] != ev.size[0]: img = cv2.resize(img, ev.size)
    for k in o['names']:
        m = ev.mask(k, t); A = {'ours': o['A_T_obj'][t, o['names'].index(k)], 'new': n['A_T_obj'][k][t]}; M = {'ours': o['meshes'][k], 'new': n['meshes'][k]}
        sd = {tn: A[tn][:3, :3] @ SLOT[tn] for tn in A}; r = dict(slot_angle_deg=float(np.degrees(np.arccos(np.clip(sd['ours'] @ sd['new'], -1, 1)))))
        for tn in A:
            for sign, lab in ((1, 'slot'), (-1, 'mirror')):
                P = strip(A[tn], centre(M[tn]), SLOT[tn], sign); uv = ev.project(P); ok_ = np.isfinite(uv).all(1)
                ui = np.round(uv[ok_]).astype(int); inb = (ui[:, 0] >= 0) & (ui[:, 0] < ev.size[0]) & (ui[:, 1] >= 0) & (ui[:, 1] < ev.size[1])
                r[f'{tn}_{lab}_cov'] = float(m[ui[inb, 1], ui[inb, 0]].mean()) if (m is not None and inb.sum() > 5) else None
                if figdir and ok and t % (stride * 4) == 0:
                    col = {('ours', 1): (0, 0, 255), ('new', 1): (255, 128, 0), ('ours', -1): (0, 0, 120), ('new', -1): (120, 60, 0)}[(tn, sign)]
                    if sign == 1:
                        for p in ui[inb]: cv2.circle(img, tuple(p), 4 if tn == 'ours' else 7, col, 1 if tn == 'new' else -1)
        row[k] = r
    if figdir and ok and t % (stride * 4) == 0:
        for k in o['names']:
            m = ev.mask(k, t)
            if m is not None:
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(img, cs, -1, (0, 255, 0), 1)
        cv2.putText(img, f'ep25 f{t}  red dots = OUR slot, orange rings = NEW slot, green = SAM2 mask', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.imwrite(f'{figdir}/ep25_slot_f{t:03d}.jpg', img[200:1400, 200:1900])
    rows.append(row)
    print(t, ' | '.join(f"{k[-1]} ang {row[k]['slot_angle_deg']:.0f} ours {row[k]['ours_slot_cov']}/{row[k]['ours_mirror_cov']} new {row[k]['new_slot_cov']}/{row[k]['new_mirror_cov']}" for k in o['names']), flush=True)
json.dump(rows, open(out, 'w'))
