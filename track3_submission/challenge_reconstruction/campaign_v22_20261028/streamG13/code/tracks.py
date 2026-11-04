"""Track sources for ep20/ep33 as kit inputs: list of (name, T[F,4,4] world_from_object, mesh_path) in slot order."""
import numpy as np, pandas as pd, os, json
from scipy.spatial.transform import Rotation as Rot
R = '/data1/SRL_V2D/challenge_reconstruction'; C = f'{R}/campaign_v22_20261028'; P = '/data1/SRL_V2D/challenge_submissions_SRL'
PKG = {'first': f'{P}/track3_all_20_episodes', 'pf6': f'{P}/track3_pot_fix_v6_20261020/all_20_episodes', 'v13': f'{P}/track3_v13_20261022/all_20_episodes',
       'v21C': f'{P}/track3_v21C_20261024/all_20_episodes', 'v22b': f'{P}/track3_v22b_20261028/all_20_episodes', 'v24': f'{P}/track3_v24_20261028/all_20_episodes',
       'v25': f'{P}/track3_v25_20261028/all_20_episodes', 'v26': f'{P}/track3_v26_20261029/all_20_episodes'}
TGT = {'v10': f'{C}/target_final_v10_g12', 'c21': f'{R}/campaign_v16_20261023/compose_target', 'c27': f'{R}/campaign_v23_20261028/target_compose27',
       'v5': f'{C}/target_final_v5', 'v6': f'{C}/target_final_v6'}
def names_of(ep):
    r = pd.read_parquet(f'{PKG["v26"]}/results/episode_{ep:06d}.parquet', columns=['object_id', 'object_body_name'])
    return [n for _, n in r.drop_duplicates('object_id').sort_values('object_id').values.tolist()]
def target(t, ep, d=None):
    d = d or f'{TGT[t]}/episode_{ep:06d}'
    return [(n, np.load(f'{d}/{n}_trajectory.npz')['T_world_obj'], f'{d}/{n}_mesh_metric.obj') for n in names_of(ep)]
def bundle(bdir, ep):
    d = f'{bdir}/episode_{ep:06d}'; p = np.load(f'{d}/object_poses.npz', allow_pickle=True); names = [str(n) for n in p['names']]
    mj = f'{d}/meshes.json'
    mp = (lambda n: f'{d}/' + json.load(open(mj))[n]['file']) if os.path.exists(mj) else (lambda n: f'{d}/assets/{n}.obj')
    return [(n, p['world_from_object'][:, names.index(n)], mp(n)) for n in names_of(ep)]
FRO = f'{R}/dexagent_remaining7_20261019/source/final_recon_ours'
def _pq(df, pre):
    T = np.tile(np.eye(4), (len(df), 1, 1))
    T[:, :3, :3] = Rot.from_quat(df[[f'{pre}_quaternion_{c}' for c in 'xyzw']].values).as_matrix()
    T[:, :3, 3] = df[[f'{pre}_position_{c}' for c in 'xyz']].values; return T
def package(p, ep, which='achieved'):
    """achieved (or reference) poses of an uploaded package on the 20 Hz reference grid (first row per reference_frame)."""
    base = PKG[p] if p in PKG else p
    r = pd.read_parquet(f'{base}/results/episode_{ep:06d}.parquet'); out = []
    for n in names_of(ep):
        s = r[r.object_body_name == n].sort_values('step').drop_duplicates('reference_frame', keep='first').sort_values('reference_frame')
        out.append((n, _pq(s, which), f'{base}/meshes/episode_{ep:06d}/{n}.obj'))
    return out
