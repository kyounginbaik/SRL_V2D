"""Kit body registration (MP-SR._register_object_body, unchanged) of candidate board meshes onto the compose meshes: rotation angle
for the official seed (1000003*25+101*slot) and the fraction of 40 other seeds giving the same answer. usage: kitreg_test.py MESH1 MESH2"""
import sys
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K, numpy as np, trimesh
from scipy.spatial.transform import Rotation as R
M = K.load_metric('MP-SR'); CT = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target/episode_000025/'
for slot, (o, p) in enumerate(zip(['wooden_piece_1', 'wooden_piece_2'], sys.argv[1:3])):
    b = trimesh.load(CT + o + '_mesh_metric.obj', force='mesh'); b.merge_vertices(); ref = (np.asarray(b.vertices, float), np.asarray(b.faces, np.int64)); anchor, _ = M._surface_moments(*ref)
    a = trimesh.load(p, force='mesh'); cand = (np.asarray(a.vertices, float), np.asarray(a.faces, np.int64))
    def ang(seed): return np.degrees(R.from_matrix(M._register_object_body(cand, ref, anchor, seed)[1]).magnitude())
    off = ang(1_000_003 * 25 + 101 * slot); others = np.array([ang(s) for s in range(1, 41)])
    print(o, 'official seed angle %.1f' % off, '| other seeds <10deg: %d/40, >170: %d/40' % ((others < 10).sum(), (others > 170).sum()))
