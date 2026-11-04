"""Kit-faithful scorer internals exposed (scoring-only; public GT enters only here).
Mirrors MP-SR: register our budgeted mesh onto the reference scan (scale-normalised ICP), convert body, anchor frame-0
scene alignment, symmetry-aware orientation. Returns the aligned tensors so errors can be decomposed."""
import sys, json, numpy as np, trimesh
from pathlib import Path
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
from v2dlb.mesh_budget import budget_mesh
from scipy.spatial.transform import Rotation as Rot
M = K.load_metric('MP-SR')
W = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical')
TUNE = [11, 12, 18, 21, 23, 24, 31, 39, 41, 44]
HELD = [0, 2, 13, 22, 26, 27, 40, 42, 43, 45]

def root(ep):
    return W / ('public' if ep in TUNE else 'public_heldout') / f'episode_{ep:06d}'

def T2p(T):
    q = Rot.from_matrix(T[..., :3, :3].reshape(-1, 3, 3)).as_quat().reshape(T.shape[:-2] + (4,))
    return np.concatenate([T[..., :3, 3], q], -1)

_c = {}
def ref_geom(name, nobj):
    key = ('r', name, nobj)
    if key not in _c:
        m = trimesh.load(K.gt_mesh_path(name), force='mesh'); m.merge_vertices()
        cap = min(40000, int((14_000_000 / nobj - 300_000) / 60))
        if len(m.faces) > cap:
            try: m = m.simplify_quadric_decimation(face_count=cap)
            except Exception: pass
        m.remove_unreferenced_vertices(); v = np.asarray(m.vertices, np.float64); f = np.asarray(m.faces, np.int64)
        a, _ = M._surface_moments(v, f); ax, o = K.symmetry_axis(name, v)
        _c[key] = (v, f, a, ax, o)
    return _c[key]

def sub_mesh(path):
    key = ('s', str(path))
    if key not in _c:
        v, f = budget_mesh(str(path), 4096, 4096); _c[key] = (np.asarray(v, np.float64), np.asarray(f, np.int64))
    return _c[key]

def gt_order(ep):
    import pyarrow.parquet as pq
    t = pq.read_table(f'{K.T3}/public/data/chunk-000/episode_{ep:06d}.parquet', columns=['observation.objects']).to_pandas()
    return [o['name'] for o in t['observation.objects'].iloc[0]], len(t)

def load_export(ep, tag):
    d = root(ep) / 'exports' / tag
    z = np.load(d / 'object_poses.npz'); names = [str(n) for n in z['names']]
    mj = json.loads((d / 'meshes.json').read_text()) if (d / 'meshes.json').exists() else {}
    meshes = {n: mj.get(n, str(root(ep) / 'objects' / n / 'mesh.obj')) for n in names}
    return dict(names=names, W=z['world_from_object'].astype(np.float64), cam=z['world_from_rgb'].astype(np.float64),
                obs=z['object_observed'] if 'object_observed' in z else None, meshes=meshes)

def score(ep, Wd, meshes, cam=None, regs=None):
    """Wd: {name: T[F,4,4] world_from_object}; meshes {name: path}. Returns dict with aligned tensors etc."""
    order, nt = gt_order(ep); objs = [o for o in order if o in Wd]
    F = min([len(Wd[o]) for o in objs] + [nt])
    A = np.stack([T2p(Wd[o][:F]) for o in objs], 1)[:, None].copy()
    G = np.stack([K.gt_world_T_obj(ep, o)[0][:F] for o in objs], 1)
    R = T2p(G)[:, None].copy()
    axes, orders, cas, rots = [], [], [], []
    for s, o in enumerate(objs):
        v, f, anc, ax, od = ref_geom(o, len(objs))
        if regs is not None and o in regs: ca, rot = regs[o]
        else: ca, rot = M._register_object_body(sub_mesh(meshes[o]), (v, f), anc, 1_000_003 * ep + 101 * s)
        A[:, :, s] = M._convert_object_body(A[:, :, s], ca, rot)
        R[:, :, s, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(R[:, :, s, 3:]), anc)
        axes.append(ax); orders.append(od); cas.append(ca); rots.append(rot)
    A0 = A.copy()
    Al = M._align_registered_scene(A, R, axes)
    # recover the scene transform: p' = S p + off
    fixed = [i for i, a in enumerate(axes) if not np.any(a)]
    if fixed or len(objs) == 1:
        an = fixed[0] if fixed else 0
        S = M._rotation_matrix(M._quaternion_multiply(R[0, 0, an, 3:], A0[0, 0, an, 3:] * [-1, -1, -1, 1]))
    else:
        ar = M._rotation_matrix(R[0, 0, 0, 3:]) @ axes[0]; aa = M._rotation_matrix(A0[0, 0, 0, 3:]) @ axes[0]
        S = M._axis_basis(ar, R[0, 0, 1, :3] - R[0, 0, 0, :3]) @ M._axis_basis(aa, A0[0, 0, 1, :3] - A0[0, 0, 0, :3]).T
    off = R[0, 0, 0, :3] - S @ A0[0, 0, 0, :3]
    pos = np.linalg.norm(Al[..., :3] - R[..., :3], axis=-1)[:, 0] * 100
    ori = np.degrees(M._symmetry_orientation_errors(Al, R, axes, orders))[:, 0]
    out = dict(objs=objs, F=F, A=Al[:, 0], R=R[:, 0], A0=A0[:, 0], S=S, off=off, axes=axes, orders=orders, cas=cas, rots=rots,
               pos=pos, ori=ori)
    if cam is not None:
        C = cam[:F]; Rc = np.einsum('ij,tjk->tik', S, C[:, :3, :3]); cc = C[:, :3, 3] @ S.T + off
        out['camR'] = Rc; out['camc'] = cc
    return out

def metrics(ep, Wd, meshes):
    """Full kit metric numbers via bscore (slow path)."""
    sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020/code/boards')
    import bscore
    return bscore.score(ep, {n: (Wd[n], meshes[n]) for n in Wd}, metrics=True)
