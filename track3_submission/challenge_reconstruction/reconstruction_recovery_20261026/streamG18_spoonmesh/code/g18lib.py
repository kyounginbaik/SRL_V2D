"""G18 library: kit-exact registration study for the wooden spoon (GT-free).
Truth variants are built ONLY from our own spoon mesh + our own GT-free ball centres (out/ballcenters.json).
No public scan is read here."""
import sys, json, numpy as np, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
M = K.load_metric('MP-SR')
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
G = RR / 'streamG18_spoonmesh'
SPOON = RR / 'selected20_final_v12b_g15/episode_000032/meshes/wooden_spoon.obj'
EPS = [3, 5, 6, 15, 16, 32, 33, 35, 36, 37]
BC = json.loads((G / 'out/ballcenters.json').read_text()) if (G / 'out/ballcenters.json').exists() else None


def load(p):
    m = trimesh.load(str(p), force='mesh', process=False)
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def components(v, f):
    m = trimesh.Trimesh(v, f, process=False)
    return sorted(m.split(only_watertight=False), key=lambda x: -x.area)


def outer(v, f):
    o = components(v, f)[0]
    return np.asarray(o.vertices, float), np.asarray(o.faces, np.int64)


def cat(*parts):
    V, F, n = [], [], 0
    for v, f in parts:
        V.append(v); F.append(f + n); n += len(v)
    return np.concatenate(V), np.concatenate(F)


def ball(c, r, sub=1):
    s = trimesh.creation.icosphere(subdivisions=sub, radius=r)
    return np.asarray(s.vertices) + c, np.asarray(s.faces, np.int64)


def balls(centres, r, sub=1):
    return cat(*[ball(np.asarray(c), r, sub) for c in centres])


def long_axis(v, f):
    c, _ = M._surface_moments(v, f)
    x = v - v.mean(0); w, U = np.linalg.eigh(x.T @ x); a = U[:, -1]
    return c, a * np.sign(a[0])


def twist_deg(R, axis):
    """swing-twist: twist angle (deg, signed) of R about axis"""
    q = Rot.from_matrix(R).as_quat(); p = np.dot(q[:3], axis)
    return float(np.degrees(2 * np.arctan2(p, q[3])) + 180) % 360 - 180


def angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def rand_rigid(seed):
    rng = np.random.default_rng(seed)
    R = Rot.random(random_state=rng.integers(1 << 31)).as_matrix(); t = rng.normal(size=3) * 0.2
    return R, t


def register(cand, truth, seed, Rt=None, tt=None, Dinv=None):
    """cand/truth: (v,f) both in OUR body frame. The truth is moved by a random rigid (Rt,tt) like an arbitrary scan frame.
    Returns body-frame rotation error (deg), twist about our long axis (deg), anchor error (mm)."""
    if Rt is None: Rt, tt = rand_rigid(seed)
    tv, tf = truth
    rv = tv @ Rt.T + tt
    anc, _ = M._surface_moments(rv, tf)
    ca, rot = M._register_object_body(cand, (rv, tf), anc, seed)
    E = Rt.T @ rot  # identity if correct (candidate body -> truth body)
    _, ax = long_axis(*cand)
    true_anc = Rt.T @ (anc - tt)  # truth anchor in truth body frame (= our frame before deformation)
    if Dinv is not None: true_anc = Dinv(true_anc)
    return dict(ang=angle_deg(E), roll=twist_deg(E, ax), anc_mm=float(np.linalg.norm(ca - true_anc) * 1000),
                E=E.tolist(), ca=ca.tolist())


# ---------------- truth variants (all built from our mesh in our body frame) ----------------
def deform(v, centre, S):
    return (v - centre) @ np.asarray(S).T + centre


def taubin(v, f, it=10):
    m = trimesh.Trimesh(v.copy(), f, process=False)
    trimesh.smoothing.filter_taubin(m, iterations=it)
    return np.asarray(m.vertices), f


def offset(v, f, d):
    m = trimesh.Trimesh(v, f, process=False)
    return v + m.vertex_normals * d, f


def keep_faces(v, f, mask):
    m = trimesh.Trimesh(v, f[mask], process=False); m.remove_unreferenced_vertices()
    return np.asarray(m.vertices), np.asarray(m.faces, np.int64)


def subdivide(v, f, n=1):
    for _ in range(n):
        v, f = trimesh.remesh.subdivide(v, f)
    return np.asarray(v), np.asarray(f, np.int64)


def centres(kind):
    if kind == 'centre': return np.array(BC['centre_arr'])
    if kind == 'front': return np.array(BC['front_median_arr'])
    if kind == 'mirror':  # chirality reversed: 180 deg roll about the long axis
        return np.array(BC['centre_arr']) * [1, -1, -1]
    if kind == 'swapLR':  # the two neck balls swapped side (y mirrored only, z kept)
        return np.array(BC['centre_arr']) * [1, -1, 1]
    if kind == 'up':  # all three lifted to +z (top face) by 6 mm, i.e. balls on the concave face
        return np.array(BC['centre_arr']) + [0, 0, 0.006]
    if kind == 'down':
        return np.array(BC['centre_arr']) - [0, 0, 0.006]
    raise KeyError(kind)


def truth_variant(name, base_v, base_f, seed=0):
    """name tokens joined by '+': geometry ops apply to the (outer) spoon, then balls are added.
    ops: self | full (keep inner shell) | s<k> uniform scale | ax<a><k> axis scale (x long, y width, z thick) |
    hz<k> handle-only thickness | tau smooth | off<mm> normal offset | noise<mm> | cutbot / cuttop / cuttip / cutbowl |
    b<kind>_<rmm> balls"""
    ov, of = outer(base_v, base_f)
    v, f = (base_v.copy(), base_f.copy()) if 'full' in name.split('+') else (ov.copy(), of.copy())
    c, _ = M._surface_moments(v, f)
    S = np.eye(3); extras = []
    rng = np.random.default_rng(seed + 17)
    for tok in name.split('+'):
        if tok in ('self', 'full'): continue
        if tok.startswith('s') and tok[1:2].isdigit():
            k = float(tok[1:]); S = S @ np.diag([k, k, k]); v = deform(v, c, np.diag([k, k, k]))
        elif tok.startswith('ax'):
            i = 'xyz'.index(tok[2]); k = float(tok[3:]); D = np.eye(3); D[i, i] = k; S = S @ D; v = deform(v, c, D)
        elif tok.startswith('hz'):
            k = float(tok[2:]); h = v[:, 0] < 0.0; v = v.copy(); v[h, 2] = (v[h, 2]) * k
        elif tok == 'tau':
            v, f = taubin(v, f, 10)
        elif tok.startswith('off'):
            v, f = offset(v, f, float(tok[3:]) / 1000)
        elif tok.startswith('noise'):
            v = v + rng.normal(size=v.shape) * float(tok[5:]) / 1000
        elif tok.startswith('cut'):
            fc = v[f].mean(1); n = trimesh.Trimesh(v, f, process=False).face_normals
            if tok == 'cutbot': mask = ~(n[:, 2] < -0.5)          # scan missed the underside
            elif tok == 'cuttop': mask = ~(n[:, 2] > 0.5)
            elif tok == 'cuttip': mask = fc[:, 0] > v[:, 0].min() + 0.02
            elif tok == 'cutbowl': mask = ~((fc[:, 0] > 0.03) & (n[:, 2] < -0.3))
            elif tok == 'cutside': mask = ~(n[:, 1] > 0.5)
            else: raise KeyError(tok)
            v, f = keep_faces(v, f, mask)
        elif tok.startswith('b'):
            kind, r = tok[1:].split('_'); r = float(r) / 1000
            cs = centres(kind)
            cs = deform(cs, c, S)  # balls ride along with a global linear deformation
            extras.append(balls(cs, r, 2))
        else:
            raise KeyError(tok)
    if extras: v, f = cat((v, f), *extras)
    Sinv = np.linalg.inv(S)
    return (v, f), (lambda p: (p - c) @ Sinv.T + c)


# ---------------- candidate meshes ----------------
def cand_variant(name, base_v, base_f, bc=None):
    """A: ours as submitted | O: outer only | B<r>: ours outer + 3 balls (centre est.) radius r mm, icosphere sub 2"""
    if name == 'A': return base_v, base_f
    if name == 'G':  # the built fix mesh, through the kit packer's budget (pads to 4096 rows)
        sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/track3_v28_20261029/all_20_episodes/submission_kit')
        from v2dlb.mesh_budget import budget_mesh
        v, f = budget_mesh(str(G / 'mesh/wooden_spoon_g18.obj'), 4096, 4096); return np.asarray(v, float), np.asarray(f, np.int64)
    ov, of = outer(base_v, base_f)
    if name == 'O': return ov, of
    if name.startswith('B'):
        r = float(name[1:].split('f')[0]) / 1000
        cs = np.array(BC['front_median_arr'] if 'f' in name[1:] else BC['centre_arr']) if bc is None else bc
        return cat((ov, of), balls(cs, r, 2))
    if name.startswith('F'):  # full (with inner shell) + balls sub 1, decimated to budget by the kit packer path
        r = float(name[1:]) / 1000
        return cat((base_v, base_f), balls(np.array(BC['centre_arr']), r, 1))
    raise KeyError(name)
