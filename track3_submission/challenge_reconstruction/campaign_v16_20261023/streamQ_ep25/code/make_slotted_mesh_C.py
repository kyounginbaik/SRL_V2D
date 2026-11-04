"""Stream Q phase 6: SLOTTED visual meshes of OUR boards = clean watertight surface of the union of our own slot-cut collision parts
(ep25_v15/collision/wooden_piece_X_cut3mm: CoACD/k-means hulls of our reconstruction meshes c7e115779dc6.. / ecf686792184.., slot cut
from our reconstructed assembled cross, +3 mm clearance). Occupancy of the union on a regular grid (point-in-convex-hull by the hulls'
plane equations), marching cubes (skimage), Taubin smoothing, quadric decimation to 20k faces (like the reconstruction meshes), same body frame as the reconstruction mesh (poses stay valid). No scan-derived geometry.
usage: make_slotted_mesh_C.py OUTDIR PITCH_M COLLISION_DIR SUFFIX"""
import sys, glob, json, hashlib, numpy as np, trimesh
from pathlib import Path
from skimage.measure import marching_cubes
C = Path(sys.argv[3]) if len(sys.argv) > 3 else Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/ep25_v15/collision')
A = Path('/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/collision_recon_slots_v3/episode_000025/assets')
SUF = sys.argv[4] if len(sys.argv) > 4 else 'cut3mm'
out = Path(sys.argv[1]); pitch = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0006; rep = {}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for o in ['wooden_piece_1', 'wooden_piece_2']:
    parts = [trimesh.load(f, force='mesh').convex_hull for f in sorted(glob.glob(str(C / f'{o}_{SUF}/part_*.obj')))]
    lo = np.min([p.bounds[0] for p in parts], 0) - 2 * pitch; hi = np.max([p.bounds[1] for p in parts], 0) + 2 * pitch
    axes = [np.arange(lo[i], hi[i] + pitch, pitch) for i in range(3)]; G = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    occ = np.zeros(len(G), bool)
    for p in parts:
        n = p.face_normals; d = -(n * p.triangles[:, 0]).sum(1); bb = np.all((G >= p.bounds[0] - 1e-9) & (G <= p.bounds[1] + 1e-9), 1); idx = np.where(bb & ~occ)[0]
        inside = np.all(G[idx] @ n.T + d <= 1e-9, 1); occ[idx[inside]] = True
    vol = occ.reshape([len(a) for a in axes]).astype(np.float32)
    v, f, _, _ = marching_cubes(np.pad(vol, 1), level=0.5, spacing=(pitch,) * 3); v = v + lo - pitch
    m = trimesh.Trimesh(v, f[:, ::-1], process=True)
    if m.volume < 0: m.invert()
    trimesh.smoothing.filter_taubin(m, iterations=10)
    m = m.simplify_quadric_decimation(face_count=20000); m.merge_vertices(); m.remove_unreferenced_vertices()
    p = out / f'{o}_visual_slotted.obj'; m.export(p)
    orig = trimesh.load(A / f'{o}_visual.obj', force='mesh')
    rep[o] = dict(path=str(p), sha256=sha(p), faces=int(len(m.faces)), watertight=bool(m.is_watertight), volume_cm3=float(m.volume * 1e6),
                  union_parts=len(parts), parts_volume_sum_cm3=float(sum(q.volume for q in parts) * 1e6), bounds=m.bounds.tolist(), orig_visual_bounds=orig.bounds.tolist(),
                  source_collision=str(C / f'{o}_{SUF}'), source_reconstruction_mesh=str(A / f'{o}_visual.obj'), source_reconstruction_sha256=sha(A / f'{o}_visual.obj'), pitch_m=pitch)
    print(o, json.dumps({k: rep[o][k] for k in ('sha256', 'faces', 'watertight', 'volume_cm3', 'parts_volume_sum_cm3')}), flush=True)
json.dump(rep, open(out / 'slotted_mesh_report.json', 'w'), indent=1)
