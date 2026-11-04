"""G18 fix mesh (GT-free): our 5fcd334c spoon with (1) the fully-internal inverted inner shell removed (630 v / 1256 f, all
vertices 0.5-5.8 mm inside the outer surface, negative volume) and (2) three mocap balls added at OUR pooled ball centres
(out/ballcenters.json 'centre_arr': stereo front-surface points pushed back by r along the ray, RGB-clock v10_g12 poses, 10 eval
episodes) with OUR detection radius (median radius_px*z/f = 6.06 mm). Same body frame -> no pose changes.
Writes mesh/wooden_spoon_g18.obj (ours frame) and mesh/wooden_spoon_g18_sub15_16.obj (frame of the e6c98973 mesh submitted in ep15/16)."""
import sys, json, numpy as np, trimesh, hashlib
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/code')
from g18lib import *
bv, bf = load(SPOON)
ov, of = outer(bv, bf)
r = BC['radius_mm_median'] / 1000
v, f = cat((ov, of), balls(np.array(BC['centre_arr']), r, 2))
out = G / 'mesh/wooden_spoon_g18.obj'
assert not out.exists()
trimesh.Trimesh(v, f, process=False).export(out)
fr = json.loads((G / 'out/subframe.json').read_text())
T = np.array(fr['15']['T_sub_from_ours']); assert np.allclose(T, fr['16']['T_sub_from_ours'])
trimesh.Trimesh(v @ T[:3, :3].T + T[:3, 3], f, process=False).export(G / 'mesh/wooden_spoon_g18_sub15_16.obj')
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/track3_v28_20261029/all_20_episodes/submission_kit')
from v2dlb.mesh_budget import budget_mesh
for p in sorted((G / 'mesh').glob('*.obj')):
    m = trimesh.load(p, force='mesh', process=False); bvv, bff = budget_mesh(str(p), 4096, 4096)
    bm = trimesh.Trimesh(bvv, bff, process=False); bm.remove_unreferenced_vertices()
    print(p.name, 'v', len(m.vertices), 'f', len(m.faces), 'after kit budget v', len(bm.vertices), 'f', len(bm.faces),
          'md5', hashlib.md5(p.read_bytes()).hexdigest()[:8], 'extents cm', (m.extents * 100).round(2).tolist(), 'radius_mm', round(r * 1000, 2))
