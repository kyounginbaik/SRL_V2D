"""Software-rendered video of a whiteboard rollout: hand, pen, board, ink. Usage: render_w.py QPOS.npy|npz OUT LABEL [SCENE_MAP]. qpos in our scene frame at 50 Hz. Green = reference ink, orange = what this pen drew (tip on board only)."""
import sys, numpy as np, mujoco, cv2, imageio, wmetric as WM
from scipy.spatial.transform import Rotation as R
src, out, label = sys.argv[1:4]; d0 = np.load(src); Q = d0['q'] if hasattr(d0, 'files') else d0; RF = WM.RF; WW, HH = 960, 720
m = mujoco.MjModel.from_xml_path(str(WM.P / 'scene.xml')); d = mujoco.MjData(m)
geoms = []
for g in range(m.ngeom):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and (nm == 'right_object_visual' or (m.geom_group[g] == 1 and 'right_' in nm)):
        mi = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mi]:m.mesh_vertadr[mi] + m.mesh_vertnum[mi]].astype(float); f = m.mesh_face[m.mesh_faceadr[mi]:m.mesh_faceadr[mi] + m.mesh_facenum[mi]]
        geoms.append((g, v, f, (225, 130, 45) if nm == 'right_object_visual' else (120, 165, 225)))
print('geoms', len(geoms))
N, S = WM.N, WM.S; cen = S + N * .08; cen = cen + np.array([0, 0, .0]); cam = S + N * .55 + np.array([0.0, 0.0, .12]); cen = RF['tip'][RF['on']].mean(0) + N * .05
fwd = cen - cam; fwd /= np.linalg.norm(fwd); right = np.cross(fwd, [0, 0, 1.]); right /= np.linalg.norm(right); up = np.cross(right, fwd); F = 2.0 * HH; L = np.array([.3, -.5, .8]); L /= np.linalg.norm(L)
def proj(p):
    r = p - cam; z = r @ fwd; return np.stack([WW / 2 + F * (r @ right) / z, HH / 2 - F * (r @ up) / z], -1), z
ex = np.cross([0, 0, 1.], N); ex /= np.linalg.norm(ex); ey = np.cross(N, ex); cs = np.array([S + a * ex + b * ey for a, b in ((-.3, -.3), (.3, -.3), (.3, .3), (-.3, .3))]); bu = np.int32(proj(cs)[0]); ink_ref = proj(RF['tip'] - N * 0)[0]
ex2 = right - N * (right @ N); ex2 /= np.linalg.norm(ex2); ey2 = np.cross(N, ex2)
if ey2 @ up < 0: ey2 = -ey2
frames = []; n = min(len(Q), RF['n']); pen_all = Q[:n, -7:]; tip = WM.tip_of(pen_all); h = (tip - S) @ N; uvt = proj(tip)[0]
for t in range(0, n, 2):
    im = np.full((HH, WW, 3), 205, np.uint8); cv2.fillConvexPoly(im, bu, (238, 238, 232), lineType=cv2.LINE_AA)
    for i in range(1, n):  # full reference ink in light green
        if RF['on'][i] and RF['on'][i - 1]: cv2.line(im, tuple(np.int32(ink_ref[i - 1])), tuple(np.int32(ink_ref[i])), (150, 215, 150), 3, cv2.LINE_AA)
    for i in range(1, t + 1):
        if abs(h[i] - WM.H0) < .005 and abs(h[i - 1] - WM.H0) < .005: cv2.line(im, tuple(np.int32(uvt[i - 1])), tuple(np.int32(uvt[i])), (205, 80, 20), 2, cv2.LINE_AA)
    d.qpos[:] = Q[t]; mujoco.mj_kinematics(m, d); tris = []
    for g, v, f, c in geoms:
        w = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]; a, b, e = w[f[:, 0]], w[f[:, 1]], w[f[:, 2]]; nr = np.cross(b - a, e - a); nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-9; uv, _ = proj(w); dp = ((a + b + e) / 3 - cam) @ fwd; sh = .35 + .65 * np.abs(nr @ L); P = np.int32(uv[f])
        tris += [(dp[i], P[i], tuple(int(x * sh[i]) for x in c)) for i in range(len(f))]
    tris.sort(key=lambda x: -x[0])
    for _, p, c in tris: cv2.fillConvexPoly(im, p, c, lineType=cv2.LINE_AA)
    pn = np.full((230, 340, 3), 245, np.uint8); cc = RF['tip'][RF['on']].mean(0); to = lambda p: np.int32(np.stack([170 + 26. * 100 * ((p - cc) @ ex2), 115 - 26. * 100 * ((p - cc) @ ey2)], -1))
    ur, ut = to(RF['tip']), to(tip)
    for i in range(1, n):
        if RF['on'][i] and RF['on'][i - 1]: cv2.line(pn, tuple(ur[i - 1]), tuple(ur[i]), (130, 205, 130), 3, cv2.LINE_AA)
    for i in range(1, t + 1):
        if abs(h[i] - WM.H0) < .005 and abs(h[i - 1] - WM.H0) < .005: cv2.line(pn, tuple(ut[i - 1]), tuple(ut[i]), (205, 80, 20), 2, cv2.LINE_AA)
    cv2.rectangle(pn, (0, 0), (339, 229), (90, 90, 90), 1); cv2.putText(pn, 'ink: green human, orange this run', (6, 222), cv2.FONT_HERSHEY_SIMPLEX, .42, (40, 40, 40), 1, cv2.LINE_AA); im[46:276, WW - 346:WW - 6] = pn
    te = np.linalg.norm(tip[t] - RF['tip'][t]) * 100; cv2.rectangle(im, (0, 0), (WW, 40), (255, 255, 255), -1); cv2.putText(im, f'{label[:70]}  t={t * .02:4.1f}s  tip err {te:5.1f} cm', (8, 27), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1, cv2.LINE_AA); frames.append(im)
imageio.mimsave(out, frames, fps=25, macro_block_size=1); print('VIDEO', out, len(frames))
