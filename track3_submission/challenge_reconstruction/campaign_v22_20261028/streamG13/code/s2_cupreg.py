"""SCORER-SIDE ONLY. Consistency of the kit registration of the ep20 package cup mesh (277d45fc, used by every upload) onto the public
blue-cup scan, vs our validated cup mesh 415ade56 (public cup clips pass with it): axis mapping and anchor along the axis."""
import numpy as np, trimesh
import kit; M = kit.M; K = kit.K
P20 = '/data1/SRL_V2D/challenge_submissions_SRL/track3_v26_20261029/all_20_episodes/meshes/episode_000020/blue_cup.obj'
P415 = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v10_g12/episode_000020/meshes/blue_cup.obj'
v, f, anc, ax, o = kit.ref_mesh(K.gt_mesh_path('blue_cup'), 'blue_cup')
def axis_of(path):
    m = trimesh.load(path, force='mesh'); return K.symmetry_axis('blue_cup', np.asarray(m.vertices))[0], m
for name, p in (('277d45fc(ep20 pkg)', P20), ('415ade56(v10/public)', P415)):
    a_c, m = axis_of(p); cand = kit.sub_mesh(p)
    ca, R = M._register_object_body(cand, (v, f), anc, 1_000_003 * 20)
    # open end: the end of the axis with the larger rim radius
    vv = np.asarray(m.vertices); c = vv.mean(0); s = (vv - c) @ a_c
    def rad(sel): q = vv[sel] - vv[sel].mean(0); return np.linalg.norm(q - np.outer(q @ a_c, a_c), axis=1).mean()
    open_c = a_c if rad(s > np.percentile(s, 95)) > rad(s < np.percentile(s, 5)) else -a_c
    sv = (v - v.mean(0)) @ ax; qv = lambda sel: np.linalg.norm((v[sel] - v[sel].mean(0)) - np.outer((v[sel] - v[sel].mean(0)) @ ax, ax), axis=1).mean()
    open_r = ax if qv(sv > np.percentile(sv, 95)) > qv(sv < np.percentile(sv, 5)) else -ax
    mapped = R @ open_c
    print(f'{name}: open-end axis mapped onto scan open-end axis: angle {np.degrees(np.arccos(np.clip(mapped @ open_r, -1, 1))):.1f} deg; candidate anchor {np.round(ca*100,2)} cm (cand surface centroid {np.round(M._surface_moments(*cand)[0]*100,2)})')
