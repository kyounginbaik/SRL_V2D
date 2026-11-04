"""Common loaders: every ep20/ep33 track expressed in frame-0 RGB camera coordinates (cam0), with each source's own mesh."""
import numpy as np, trimesh, json, os
R = '/data1/SRL_V2D/challenge_reconstruction'
SRC = {
 'v10': f'{R}/reconstruction_recovery_20261026/selected20_final_v10_g12',
 'fro': f'{R}/dexagent_remaining7_20261019/source/final_recon_ours',   # old target for ep20 (14,15,16,20,25,28,30)
 'c21': f'{R}/campaign_v16_20261023/compose_target',
 'c27': f'{R}/campaign_v23_20261028/target_compose27',
}
def ep_dir(src, ep): return f'{SRC[src]}/episode_{ep:06d}'
def load(src, ep):
    """-> dict name -> (T_cam0_obj [F,4,4], mesh vertices (body frame))"""
    d = ep_dir(src, ep); out = {}
    if src in ('v10', 'fro') or os.path.exists(f'{d}/object_poses.npz') and src not in ('c21','c27'):
        p = np.load(f'{d}/object_poses.npz', allow_pickle=True)
        W0 = np.linalg.inv(p['world_from_rgb'][0].astype(float))
        for k, n in enumerate(p['names']):
            n = str(n)
            if src == 'v10':
                mf = f'{d}/' + json.load(open(f'{d}/meshes.json'))[n]['file']
            else:
                mf = f'{d}/assets/{n}.obj'
            out[n] = (W0 @ p['world_from_object'][:, k], trimesh.load(mf, force='mesh').vertices)
    else:
        cm = np.load(f'{d}/camera_motion.npz'); A = cm['T_a_from_rectL']
        for f in sorted(os.listdir(d)):
            if f.endswith('_trajectory.npz'):
                n = f[:-len('_trajectory.npz')]
                T = np.load(f'{d}/{f}')['T_world_obj']
                out[n] = (A @ T, trimesh.load(f'{d}/{n}_mesh_metric.obj', force='mesh').vertices)
    return out
