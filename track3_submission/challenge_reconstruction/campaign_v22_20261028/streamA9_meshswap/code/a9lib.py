"""A9 shared helpers: load v6 target poses/meshes, component reference/achieved poses (GT-free)."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.transform import Rotation as R
ROOT = Path('/data1/SRL_V2D/challenge_reconstruction')
CAMP = ROOT / 'campaign_v22_20261028'
V6 = ROOT / 'reconstruction_recovery_20261026/selected20_final_v6'
TGT = CAMP / 'target_final_v6'
CAN = ROOT / 'reconstruction_recovery_20261026/canonical/evaluation'
PKG = Path('/data1/SRL_V2D/challenge_submissions_SRL/track3_v23b_20261028/all_20_episodes')
B2 = ROOT / 'campaign_v16_20261023/common_v2_candidates'
SEL = json.loads((CAMP / 'selection_v23b.json').read_text())

def ename(ep): return f'episode_{int(ep):06d}'

def v6_poses(ep):
    z = np.load(V6 / ename(ep) / 'object_poses.npz'); return [str(n) for n in z['names']], z

def tgt_traj(ep, obj): return np.load(TGT / ename(ep) / f'{obj}_trajectory.npz')['T_world_obj']

def comp_dir(ep): return B2 / SEL[str(int(ep))] / ename(ep)

def comp_poses(ep, obj, which='reference'):
    r = pd.read_parquet(comp_dir(ep) / 'results' / f'{ename(ep)}.parquet'); r = r[(r.environment_id == r.environment_id.min()) & (r.object_body_name == obj)].sort_values('step')
    p = r[[f'{which}_position_{a}' for a in 'xyz']].values; q = r[[f'{which}_quaternion_{a}' for a in 'xyzw']].values
    T = np.tile(np.eye(4), (len(p), 1, 1)); T[:, :3, :3] = R.from_quat(q).as_matrix(); T[:, :3, 3] = p; return T

def inv(T):
    Ti = np.tile(np.eye(4), T.shape[:-2] + (1, 1)); Rt = np.swapaxes(T[..., :3, :3], -1, -2); Ti[..., :3, :3] = Rt; Ti[..., :3, 3] = -np.einsum('...ij,...j->...i', Rt, T[..., :3, 3]); return Ti

def rot_mean(Rs):
    q = R.from_matrix(Rs).mean(); return q.as_matrix()

def angdeg(A, B): return float(np.degrees(R.from_matrix(A @ B.T).magnitude()))

def comp_component(ep):
    return Path(json.loads((comp_dir(ep) / 'completed.json').read_text())['legacy_component'])

def src_poses(ep):
    c = comp_component(ep) / 'portable/source' / ename(ep); z = np.load(c / 'object_poses.npz'); return [str(n) for n in z['names']], z, c
