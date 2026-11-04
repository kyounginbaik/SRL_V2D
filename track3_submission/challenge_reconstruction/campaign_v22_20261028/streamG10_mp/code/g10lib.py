"""G10: kit-faithful per-frame errors of a submitted package (or a recon) against MANY candidate truths, with the
registration done once. Mirrors streamG9_gross/code/perframe.py (kit seed, anchor frame-0 alignment, symmetry-aware ori)."""
import sys, os, glob, json
import numpy as np, pandas as pd, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
from scipy.spatial.transform import Rotation as Rot
M = K.load_metric('MP-SR')
C = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028'
RR = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026'
PKG = {'v24': '/data1/SRL_V2D/challenge_submissions_SRL/track3_v24_20261028/all_20_episodes',
       'v25': '/data1/SRL_V2D/challenge_submissions_SRL/track3_v25_20261028/all_20_episodes'}


def loadmesh(p, cap=40000):
    m = trimesh.load(p, force='mesh'); m.merge_vertices()
    if len(m.faces) > cap:
        try: m = m.simplify_quadric_decimation(face_count=cap)
        except Exception: pass
    m.remove_unreferenced_vertices(); return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def load_sub(src, ep):
    """-> names (kit slot order), list of (nf,4,4) world poses, list of candidate meshes"""
    r = pd.read_parquet(f'{src}/results/episode_{ep:06d}.parquet')
    names = [n for _, n in r.drop_duplicates('object_id').sort_values('object_id')[['object_id', 'object_body_name']].values.tolist()]
    sub = pd.read_parquet(f'{src}/track3_submission.parquet'); sub = sub[sub.row_id.str.contains(f'/e{ep:06d}/')]
    Ts, cands = [], []
    for slot, n in enumerate(names):
        s = sub[sub.row_id.str.endswith(f'/o{slot}')].sort_values('row_id')
        T = np.tile(np.eye(4), (len(s), 1, 1)); T[:, :3, :3] = Rot.from_quat(s[['quat_x', 'quat_y', 'quat_z', 'quat_w']].values).as_matrix()
        T[:, :3, 3] = s[['pos_x', 'pos_y', 'pos_z']].values; Ts.append(T)
        enc = s.object_mesh_b64.fillna('').astype(str); cands.append(M._decode_object_mesh(enc[enc.str.strip() != ''].iloc[0]))
    return names, Ts, cands


def load_bundle(bundle, ep, names):
    d = np.load(f'{bundle}/episode_{ep:06d}/object_poses.npz'); bn = [str(n) for n in d['names']]
    meshes = json.loads(open(f'{bundle}/episode_{ep:06d}/meshes.json').read())
    Ts = [d['world_from_object'][:, bn.index(n)] for n in names]
    ms = [loadmesh(f'{bundle}/episode_{ep:06d}/' + meshes[n]['file'], 1 << 30) for n in names]
    return Ts, ms, d


def load_target(tdir, ep, names):
    Ts = [np.load(f'{tdir}/episode_{ep:06d}/{n}_trajectory.npz')['T_world_obj'] for n in names]
    ms = [loadmesh(f'{tdir}/episode_{ep:06d}/{n}_mesh_metric.obj') for n in names]
    return Ts, ms


class Scorer:
    """registration of the submission meshes onto the reference meshes (kit seed), once."""
    def __init__(self, ep, names, cands, refmeshes, seed_off=0):
        self.ep, self.names = ep, names
        self.anchor, self.axes, self.orders, self.regs, self.refmeshes = [], [], [], [], refmeshes
        for slot, n in enumerate(names):
            rv, rf = refmeshes[slot]; a, _ = M._surface_moments(rv, rf)
            ax, o = K.symmetry_axis(n, rv)
            ca, R = M._register_object_body(cands[slot], (rv, rf), a, 1_000_003 * ep + 101 * slot + seed_off)
            self.anchor.append(a); self.axes.append(ax); self.orders.append(o); self.regs.append((ca, R))

    @staticmethod
    def q(T):
        return np.concatenate([T[:, :3, 3], Rot.from_matrix(T[:, :3, :3]).as_quat()], 1)

    def errors(self, subT, refT):
        nf = min(min(len(t) for t in subT), min(len(t) for t in refT))
        ach = np.stack([self.q(t[:nf]) for t in subT], 1)[:, None]; ref = np.stack([self.q(t[:nf]) for t in refT], 1)[:, None]
        for slot in range(len(self.names)):
            ca, R = self.regs[slot]
            ach[:, :, slot] = M._convert_object_body(ach[:, :, slot], ca, R)
            ref[:, :, slot, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, slot, 3:]), self.anchor[slot])
        al = M._align_registered_scene(ach, ref, self.axes)
        pos = np.linalg.norm(al[..., :3] - ref[..., :3], axis=-1)[:, 0] * 100
        ori = np.degrees(M._symmetry_orientation_errors(al, ref, self.axes, self.orders))[:, 0]
        return pos, ori

    def summary(self, subT, refT):
        pos, ori = self.errors(subT, refT)
        pm, om = pos.mean(0), ori.mean(0)
        mp = bool(np.all((pm < 3) & (om < 30)))
        sp = bool(pm.mean() <= 10 and om.mean() <= np.degrees(0.5))
        return dict(objects={n: [round(float(pm[i]), 2), round(float(om[i]), 1)] for i, n in enumerate(self.names)}, MP=mp, SP=sp,
                    sp_mean_ori=round(float(om.mean()), 1)), pos, ori


def roll_axis(v, f):
    """long (major PCA) axis + centroid of a mesh"""
    c, _ = M._surface_moments(v, f)
    x = v - v.mean(0); w, U = np.linalg.eigh(x.T @ x)
    return c, U[:, -1]


def body_roll(T, c, axis, ang_deg):
    """T @ rotation about the body axis through centroid c by ang (per frame array)"""
    ang = np.broadcast_to(np.asarray(ang_deg, float), (len(T),))
    out = T.copy()
    for f in range(len(T)):
        if ang[f] == 0: continue
        R = Rot.from_rotvec(np.radians(ang[f]) * axis).as_matrix()
        X = np.eye(4); X[:3, :3] = R; X[:3, 3] = c - R @ c
        out[f] = T[f] @ X
    return out
