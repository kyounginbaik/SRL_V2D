"""Stream Q phase 6: cut an assembly slot out of OUR validated (uncut) collision parts of a board, in the board BODY frame:
slot = {|x - XS| < W/2, z*SIGN > ZC*SIGN} (opens towards SIGN*z), all y. Convex parts are clipped by the slot planes (same clip
as ep25_v15/code/cut_slot_parts.py); parts outside the slot are untouched. usage: cut_slot_body.py BOARD XS W SIGN OUTDIR [ZC]"""
import sys, glob, json, numpy as np, trimesh
from pathlib import Path
A = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/collision_recon_slots_v3/episode_000025/assets')
board, XS, W, SIGN, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5]); ZC = float(sys.argv[6]) if len(sys.argv) > 6 else 0.
def clip(mesh, nrm, point):
    nrm = np.asarray(nrm, float); d = (mesh.vertices - point) @ nrm; keep = list(mesh.vertices[d >= -1e-12])
    for a, b in mesh.edges_unique:
        da, db = d[a], d[b]
        if (da < 0) != (db < 0) and abs(da - db) > 1e-15: t = da / (da - db); keep.append(mesh.vertices[a] + t * (mesh.vertices[b] - mesh.vertices[a]))
    keep = np.array(keep)
    if len(keep) < 4: return None
    try: h = trimesh.convex.convex_hull(keep)
    except Exception: return None
    return h if h.volume > 1e-9 else None
lo, hi = XS - W / 2, XS + W / 2; parts = [trimesh.load(f, force='mesh').convex_hull for f in sorted(glob.glob(str(A / f'{board}_convex_repaired/part_*.obj')))]; newp = []
for p in parts:
    x = p.vertices[:, 0]; z = p.vertices[:, 2] * SIGN
    if x.max() <= lo or x.min() >= hi or z.max() <= ZC * SIGN: newp.append(p); continue
    pieces = [clip(p, [-1, 0, 0], [lo, 0, 0]), clip(p, [1, 0, 0], [hi, 0, 0])]
    mid = clip(p, [1, 0, 0], [lo, 0, 0]); mid = clip(mid, [-1, 0, 0], [hi, 0, 0]) if mid is not None else None
    pieces.append(clip(mid, [0, 0, -SIGN], [0, 0, ZC]) if mid is not None else None)
    newp += [q.convex_hull for q in pieces if q is not None and len(q.faces) and q.volume > 1e-9]
out.mkdir(parents=True, exist_ok=False)
for k, p in enumerate(newp): p.export(out / f'part_{k:03d}.obj')
rep = dict(board=board, slot_centre_x_m=XS, slot_width_m=W, opens='+z' if SIGN > 0 else '-z', z_from_m=ZC, parts_before=len(parts), parts_after=len(newp), source=str(A / f'{board}_convex_repaired'))
(out / 'cut_report.json').write_text(json.dumps(rep, indent=1)); print(json.dumps(rep))
