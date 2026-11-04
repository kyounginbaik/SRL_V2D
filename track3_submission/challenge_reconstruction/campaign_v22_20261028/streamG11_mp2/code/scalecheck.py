"""SCORER-SIDE diagnostic only: kit registration scale (scan surface radius / our surface radius) per mesh type."""
import sys
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K, trimesh, numpy as np
M = K.load_metric('MP-SR'); C = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical'
def L(p):
    m = trimesh.load(p, force='mesh'); m.merge_vertices(); return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
for obj, ours in [('beige_cup', f'{C}/public/episode_000011/objects/beige_cup/mesh.obj'), ('blue_cup', f'{C}/public/episode_000011/objects/blue_cup/mesh.obj'),
                  ('water_pitcher', f'{C}/public/episode_000021/objects/water_pitcher/mesh.obj'), ('white_pot_lid', f'{C}/public_heldout/episode_000000/objects/white_pot_lid/mesh.obj'),
                  ('white_pot', f'{C}/public_heldout/episode_000000/objects/white_pot/mesh.obj')]:
    s = L(K.gt_mesh_path(obj)); o = L(ours); _, rs = M._surface_moments(*s); _, ro = M._surface_moments(*o)
    print(obj, 'scale scan/ours %.3f' % (rs / ro), 'extents ours', (o[0].max(0) - o[0].min(0)).round(3), flush=True)
