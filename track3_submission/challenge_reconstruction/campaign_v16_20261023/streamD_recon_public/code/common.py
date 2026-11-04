"""Stream D: load the best public-clip tracks (final recipes of public_gt_check_20261020) and score them with the kit's own
registration + anchor alignment, exposing the registered tensors so anchor/bias experiments can be run on top."""
import sys, os, glob, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
W = '/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020'
for p in (W + '/code', W + '/code/postprocess', W + '/code/sweeper', W + '/code/boards', W + '/cup'): sys.path.insert(0, p)
import score_public_tracking as S, evaluate2 as E2, pp
from scipy.spatial.transform import Rotation as R
K = S.K
MESH = lambda o: f'{W}/meshes/{o}.obj'
CLIPS = {41: ['white_pot'], 44: ['white_pot'], 39: ['mini_dust_pan'], 18: ['mini_sweeper', 'mini_dust_pan'], 31: ['mini_sweeper'],
         11: ['blue_cup', 'beige_cup'], 12: ['blue_cup'], 21: ['blue_cup'], 23: ['wooden_piece_1', 'wooden_piece_2'], 24: ['wooden_piece_1', 'wooden_piece_2']}
_M = {}
def mods():
    if not _M: _M.update({n: K.load_metric(n) for n in ['AUC', 'MP-SR']})
    return _M

def final_tracks(ep):
    """{obj: T[n,4,4]} world-from-object (our world), recipe per object type; plus c2w[n,4,4] (our world from rgb camera)."""
    out = {}; c2w = None
    for obj in CLIPS[ep]:
        if obj in ('wooden_piece_1', 'wooden_piece_2'):
            z = np.load(f'{W}/boards/episode_{ep:06d}/final_tracks.npz'); names = [str(x) for x in z['names']]
            out[obj] = z['world_from_object'][:, names.index(obj)].astype(float); c2w = z['world_from_rgb'].astype(float); continue
        A = S.stageA_poses(ep, obj); c2w = A['c2w'].astype(float) if c2w is None else c2w
        tw, c, w = pp.principal_twins(MESH(obj))
        if obj in ('white_pot', 'mini_dust_pan'):
            f = f'{W}/stereo/episode_{ep:06d}/pp_{obj}_metric.npz'
            T = np.load(f)['polish'].astype(float) if os.path.exists(f) else E2.stageB_tag(ep, obj, '_metric')
            T = pp.smooth(T, c, half=2)
        elif obj == 'mini_sweeper':
            import eval_face as EF
            T = E2.stageB_tag(ep, obj, '_metric'); pls = EF.scorer_placements_mesh(MESH(obj), ep, 0); T = EF.relabel(T, pls, 3)[0]   # grey-ball face choice (3 on both clips)
        elif obj == 'blue_cup':
            import cup_continuity as CC, trimesh
            T = E2.stageB_tag(ep, obj, '_metric'); T = CC.fix(T, np.asarray(trimesh.load(MESH(obj), force='mesh').vertices, float))[0]
        else:
            T = E2.stageB_tag(ep, obj, '_metric')
        out[obj] = np.asarray(T, float)
    n = min([len(t) for t in out.values()] + [len(c2w)])
    return {o: t[:n] for o, t in out.items()}, c2w[:n]

class Scene:
    """Registered tensors for one clip: achieved (our, registered to the scan body frame, NOT yet aligned), reference (GT, anchored)."""
    def __init__(self, ep, tracks, order=None):
        M = mods()['MP-SR']; self.M = M; self.ep = ep
        import pyarrow.parquet as pq
        t = pq.read_table(f'{K.T3}/public/data/chunk-000/episode_{ep:06d}.parquet', columns=['observation.objects']).to_pandas()
        gt_order = [o['name'] for o in t['observation.objects'].iloc[0]]
        self.names = [o for o in (order or gt_order) if o in tracks]
        n = min([len(tracks[o]) for o in self.names] + [len(t)]); self.n = n
        from v2dlb.mesh_budget import budget_mesh
        geometry = K.reference_geometry(self.names, mods()['AUC'])
        ach = np.zeros((n, 1, len(self.names), 7)); ref = np.zeros((n, 1, len(self.names), 7)); axes, orders = [], []
        self.gt_vis = {}
        for slot, obj in enumerate(self.names):
            gt, vis = K.gt_world_T_obj(ep, obj); self.gt_vis[obj] = vis[:n]
            qg = R.from_matrix(gt[:n, :3, :3]).as_quat(); qo = R.from_matrix(tracks[obj][:n, :3, :3]).as_quat()
            ref[:, 0, slot, :3] = gt[:n, :3, 3]; ref[:, 0, slot, 3:] = qg; ach[:, 0, slot, :3] = tracks[obj][:n, :3, 3]; ach[:, 0, slot, 3:] = qo
            mv, mf = budget_mesh(MESH(obj), 4096, 4096); cand = (np.asarray(mv, float), np.asarray(mf, np.int64))
            mesh = (geometry[f'v{slot}'], geometry[f'f{slot}'].astype(np.int64)); anchor = geometry[f'anchor{slot}']
            ca, rot = M._register_object_body(cand, mesh, anchor, 1_000_003 * ep + 101 * slot)
            ach[:, :, slot] = M._convert_object_body(ach[:, :, slot], ca, rot)
            ref[:, :, slot, :3] += np.einsum('...ij,j->...i', M._rotation_matrix(ref[:, :, slot, 3:]), anchor)
            axes.append(geometry[f'axis{slot}']); orders.append(int(geometry[f'order{slot}']))
        self.ach, self.ref, self.axes, self.orders = ach, ref, axes, orders
        fixed = [i for i, a in enumerate(axes) if not np.any(a)]
        self.anchor = fixed[0] if fixed else 0
    def evaluate(self, ach=None):
        """Align like the scorer and return per-object mean pos cm / ori deg, per-frame arrays, and the aligned tensor."""
        M = self.M; a = self.ach if ach is None else ach
        aligned = M._align_registered_scene(a, self.ref, self.axes)
        pos = np.linalg.norm(aligned[..., :3] - self.ref[..., :3], axis=-1)[:, 0] * 100
        ori = np.degrees(M._symmetry_orientation_errors(aligned, self.ref, self.axes, self.orders))[:, 0]
        return {o: (float(pos[:, i].mean()), float(ori[:, i].mean())) for i, o in enumerate(self.names)}, pos, ori, aligned
    def align_rotation(self, ach=None):
        a = self.ach if ach is None else ach; aligned = self.M._align_registered_scene(a, self.ref, self.axes)
        Rb = R.from_quat(a[0, 0, self.anchor, 3:]).as_matrix(); Ra = R.from_quat(aligned[0, 0, self.anchor, 3:]).as_matrix()
        return Ra @ Rb.T

def perturb_frame0(ach, slot, rotvec_world=None, dpos=None):
    """Perturb only frame 0 of one object (registered tensor): rotate about its own centre by rotvec (world axes), shift by dpos (m)."""
    a = ach.copy(); p = a[0, 0, slot]
    if rotvec_world is not None:
        Rp = R.from_rotvec(rotvec_world); a[0, 0, slot, 3:] = (Rp * R.from_quat(p[3:])).as_quat()
    if dpos is not None: a[0, 0, slot, :3] = p[:3] + np.asarray(dpos)
    return a

def static_window(T, c, speed_thr=0.01, min_len=5, fps=20.):
    cen = np.einsum('nij,j->ni', T[:, :3, :3], c) + T[:, :3, 3]; v = np.linalg.norm(np.diff(cen, axis=0), axis=1) * fps
    from scipy.ndimage import uniform_filter1d
    vs = uniform_filter1d(v, 5); moving = np.flatnonzero(vs > speed_thr)
    k = int(moving[0]) + 1 if len(moving) else len(T)
    return k if k >= min_len else 0

def mean_pose(T):
    q = R.from_matrix(T[:, :3, :3]).as_quat(); q[np.einsum('ij,j->i', q, q[0]) < 0] *= -1
    qm = q.mean(0); qm /= np.linalg.norm(qm); out = np.eye(4); out[:3, :3] = R.from_quat(qm).as_matrix(); out[:3, 3] = np.median(T[:, :3, 3], axis=0); return out
