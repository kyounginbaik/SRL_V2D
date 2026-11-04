"""Resting-contact diagnostic in the setup (table z=0) frame: our reference vs compose (G0 = pot frame-0 alignment).
usage: rest_diag.py EP RESULTS_PARQUET EPISODE_DIR MESH_DIR"""
import sys, json, numpy as np, trimesh
sys.path.insert(0, __file__.rsplit('/', 1)[0]); import tlib as L
ep, res, epd, md = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
S = np.array(json.load(open(epd + '/input_manifest.json'))['setup_from_source'])
Tc, X = L.compose_ourbody(ep); A, B = L.results_tracks(res)
V = {o: np.asarray(trimesh.load(f'{md}/{o}.obj', force='mesh').vertices) for o in L.NAMES}
G0 = B['white_pot'][0] @ np.linalg.inv(Tc['white_pot'][0])
n = len(B['white_pot'])
def low(T, o): P = V[o] @ T[:3, :3].T + T[:3, 3]; return P[:, 2].min() * 100
for f in [0, 10, 20, n // 2, n - 20, n - 1]:
    s = ' '.join(f"{o[:5]} ref {low(S @ B[o][f], o):6.2f} cmp {low(S @ G0 @ Tc[o][f], o):6.2f}" for o in L.NAMES)
    print(f'f{f:3d} lowest z (cm, setup): {s}')
# pot axis tilt: angle between pot body axis (mesh principal axis) and setup z
v = V['white_pot'] - V['white_pot'].mean(0); w, U = np.linalg.eigh(v.T @ v); ax = U[:, np.argmin(np.abs(w - np.median(w)) * 0 + [abs(w[0]-w[1])+abs(w[0]-w[2]), abs(w[1]-w[0])+abs(w[1]-w[2]), abs(w[2]-w[0])+abs(w[2]-w[1])]) * 0 + 0] if False else None
for f in [0, n - 1]:
    for lab, T in (('ref', S @ B['white_pot'][f]), ('cmp', S @ G0 @ Tc['white_pot'][f])):
        P = V['white_pot'] @ T[:3, :3].T + T[:3, 3]; b = P[P[:, 2] < P[:, 2].min() + 0.004]
        c = b - b.mean(0); nrm = np.linalg.eigh(c.T @ c)[1][:, 0]
        print(f'f{f} {lab} pot bottom-ring normal tilt vs setup z: {np.degrees(np.arccos(abs(nrm[2]))):.1f} deg, ring pts {len(b)}, ring z spread {np.ptp(b[:,2])*1000:.1f} mm')
# plane fit through resting contacts of the compose scene (G0 setup frame): pot ring centroid f0/end (weight 1 each), spoon lowest point f0/end
pts, wts = [], []
for f in [0, n - 1]:
    for o in L.NAMES:
        T = S @ G0 @ Tc[o][f]; P = V[o] @ T[:3, :3].T + T[:3, 3]
        k = P[:, 2] < P[:, 2].min() + (0.004 if o == 'white_pot' else 0.002); pts.append(P[k].mean(0)); wts.append(1.)
pts = np.array(pts); A_ = np.c_[pts[:, :2], np.ones(len(pts))]; coef = np.linalg.lstsq(A_, pts[:, 2], rcond=None)[0]
nrm = np.r_[-coef[0], -coef[1], 1]; nrm /= np.linalg.norm(nrm)
print('contacts (x,y,z cm):', np.round(pts * 100, 2).tolist())
print(f'plane tilt {np.degrees(np.arccos(nrm[2])):.2f} deg, residuals mm', np.round((pts[:, 2] - A_ @ coef) * 1000, 1).tolist())
