"""K35 helpers: kit-faithful aligned errors between two pose tracks with the SAME mesh (registration = identity), using the
kit's own _align_registered_scene / _symmetry_orientation_errors (MP-SR.py). GT-free: both tracks are ours."""
import sys, numpy as np, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
from scipy.spatial.transform import Rotation as Rot
M = K.load_metric('MP-SR')

def mesh_info(path, name):
    m = trimesh.load(path, force='mesh'); m.merge_vertices()
    v = np.asarray(m.vertices, float); f = np.asarray(m.faces, np.int64)
    anchor, _ = M._surface_moments(v, f); ax, o = K.symmetry_axis(name, v)
    c = v - v.mean(0); w, U = np.linalg.eigh(c.T @ c / len(c))
    return dict(v=v, f=f, anchor=anchor, axis=ax, order=o, pca=U[:, ::-1], points=M._sample_mesh_surface(v, f, 4000, 0) - anchor)

def tensor(tracks, infos):
    T = len(tracks[0]); out = np.zeros((T, 1, len(tracks), 7))
    for k, (tr, inf) in enumerate(zip(tracks, infos)):
        out[:, 0, k, :3] = tr[:, :3, 3] + np.einsum('tij,j->ti', tr[:, :3, :3], inf['anchor'])
        out[:, 0, k, 3:] = Rot.from_matrix(tr[:, :3, :3]).as_quat()
    return out

def aligned_errors(ach_tracks, ref_tracks, infos):
    a = tensor(ach_tracks, infos); r = tensor(ref_tracks, infos); axes = [i['axis'] for i in infos]; orders = [i['order'] for i in infos]
    a = M._align_registered_scene(a, r, axes)
    pos = np.linalg.norm(a[..., :3] - r[..., :3], axis=-1)[:, 0] * 100
    ori = np.degrees(M._symmetry_orientation_errors(a, r, axes, orders))[:, 0]
    return pos, ori   # (T, K) per frame

def body_rot(tr, inf, rotvec_body):
    """Apply a constant body-frame rotation about the mesh anchor."""
    E = Rot.from_rotvec(rotvec_body).as_matrix(); c = inf['anchor']; B = np.eye(4); B[:3, :3] = E; B[:3, 3] = c - E @ c
    return tr @ B
