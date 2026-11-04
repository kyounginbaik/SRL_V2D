"""Board slot finder: in the board's own PCA frame (u=long, w=width, n=thin), rasterise dense surface samples onto the (u,w)
plane at 2 mm; the slot is the empty run of cells inside the outline near the board centre. Returns the slot mouth direction
in BODY coordinates (unit vector along +-w pointing from the board centre towards the open end) and the slot cell centres."""
import numpy as np, trimesh
def slot_dir(mesh, cell=0.002, verbose=False):
    v = np.asarray(mesh.vertices, float); c = v.mean(0); w_, U = np.linalg.eigh(np.cov((v - c).T)); U = U[:, ::-1]   # u, w, n
    P = trimesh.sample.sample_surface(mesh, 400000, seed=1)[0]; q = (P - c) @ U
    lo, hi = q.min(0), q.max(0); nu = int((hi[0] - lo[0]) / cell) + 1; nw = int((hi[1] - lo[1]) / cell) + 1
    occ = np.zeros((nu, nw), int); iu = ((q[:, 0] - lo[0]) / cell).astype(int); iw = ((q[:, 1] - lo[1]) / cell).astype(int); np.add.at(occ, (iu, iw), 1)
    occ = occ > 0; cu = int((0 - lo[0]) / cell)
    band = occ[max(cu - 25, 0):cu + 25]   # +-5 cm around the centre along the long axis
    emp = ~band; colfrac = emp.mean(0)   # fraction empty per width cell
    # empty cells per half
    mid = int((0 - lo[1]) / cell); a = emp[:, :mid].sum(); b = emp[:, mid:].sum()
    # slot u-position: column-wise per u row count of empty cells
    rowemp = emp.sum(1); uc = lo[0] + (max(cu - 25, 0) + np.argmax(rowemp) + .5) * cell
    d = U[:, 1] * (1 if b > a else -1)
    if verbose: print('empty cells -w half', a, '+w half', b, 'slot u', round(uc, 4))
    return d, dict(center=c, U=U, empty_minus=int(a), empty_plus=int(b), slot_u=float(uc), long=U[:, 0], normal=U[:, 2])
if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        m = trimesh.load(p, force='mesh', process=False); d, info = slot_dir(m, verbose=True); print(p.split('/')[-1], 'slot dir body', d.round(3), 'normal', info['normal'].round(3))
