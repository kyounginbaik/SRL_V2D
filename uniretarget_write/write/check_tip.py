"""Does the mocap pen tip touch the board during write_on_whiteboard? Reports pen and board geometry in z-up coordinates (same rotation as SPIDER)."""
import pickle, json, numpy as np, trimesh
RT = "/lustre/exaflops/retarget_bench/data/oakink2_root"
T = "scene_02__A002++seq__e9aab4c73ce9d2c16b85__2023-04-22-09-37-57"
PEN, BOARD = "O02@0029@00011", "O02@0054@00001"
S, E = 2109, 3474
a = pickle.load(open(f"{RT}/anno_preview/{T}.pkl", "rb"))
Z = np.eye(4); Z[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
pen = trimesh.load(f"{RT}/object_repair/align_ds/{PEN}/model.obj", force="mesh")
board = trimesh.load(f"{RT}/object_repair/align_ds/{BOARD}/model.obj", force="mesh")
print("pen extents", pen.extents.round(4), "wt", pen.is_watertight, "faces", len(pen.faces))
print("board extents", board.extents.round(4), "wt", board.is_watertight, "faces", len(board.faces))
fs = [f for f in sorted(a["obj_transf"][PEN]) if S <= f < E]
TB = Z @ np.asarray(a["obj_transf"][BOARD][fs[0]])
bv = trimesh.transform_points(board.vertices, TB)
print("board world bounds z-up", bv.min(0).round(3), bv.max(0).round(3))
# board drift
pb = np.stack([np.asarray(a["obj_transf"][BOARD][f])[:3, 3] for f in fs]); print("board drift cm", (np.ptp(pb, 0) * 100).round(3))
# Pen long axis from PCA of vertices; tip candidates are both ends.
v = np.asarray(pen.vertices); c = v.mean(0)
ax = np.linalg.svd(v - c)[2][0]
s = (v - c) @ ax
ends = {"+": v[s > s.max() - 0.003].mean(0), "-": v[s < s.min() + 0.003].mean(0)}
print("pen axis", ax.round(3), "length cm", round((s.max() - s.min()) * 100, 2))
# Board's face: thinnest extent axis in its own frame.
bi = int(np.argmin(board.extents)); print("board thin axis (local)", bi)
n_w = TB[:3, bi]; 
lo, hi = board.bounds[:, bi]
face_pts = {"lo": lo, "hi": hi}
out = {}
for k, e in ends.items():
    W = np.stack([(Z @ np.asarray(a["obj_transf"][PEN][f]) @ np.r_[e, 1])[:3] for f in fs])
    L = (W - TB[:3, 3]) @ TB[:3, :3]  # into board local
    for fk, fv in face_pts.items():
        d = L[:, bi] - fv
        print(f"end {k} face {fk}: signed dist cm min {d.min()*100:.2f} p10 {np.percentile(d,10)*100:.2f} median {np.median(d)*100:.2f} p90 {np.percentile(d,90)*100:.2f} max {d.max()*100:.2f}")
    out[k] = (W, L)
# Closest approach with full mesh sampling every 15 frames.
pq = trimesh.proximity.ProximityQuery(board)
for k, e in ends.items():
    W, L = out[k]
    dist = np.abs(pq.signed_distance(L[::15]))
    print(f"end {k}: mesh dist cm p10 {np.percentile(dist,10)*100:.2f} median {np.median(dist)*100:.2f} p90 {np.percentile(dist,90)*100:.2f}; frac under .5cm {(dist<.005).mean():.2f} under 1cm {(dist<.01).mean():.2f}")
np.savez("tip.npz", fs=np.array(fs), tip_plus=out["+"][0], tip_minus=out["-"][0], local_plus=out["+"][1], local_minus=out["-"][1], thin=bi, face=np.array([lo, hi]))
