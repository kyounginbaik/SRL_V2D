"""compose27 (or c21) object poses mapped into OUR camera / world for one episode (all frames, no gating), as in
streamG9_gross/code/g9_spoon_adopt.py: kit registration of OUR mesh onto the compose mesh + one constant camera-frame
correction fitted on agreeing frames (< 20 deg). mode 'pose' = compose pose; 'rot' = compose rotation about OUR centroid.
-> returns dict name -> (nf,4,4) world_from_object (our world), plus per-frame angle to ours."""
import numpy as np, trimesh
from pathlib import Path
import sys
sys.path.insert(0, '/data1/SRL_V2D/challenge_submissions_SRL/analysis_20261020/scripts')
import public_kit_score as K
from scipy.spatial.transform import Rotation as Rot
M = K.load_metric('MP-SR')
RR = Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
TDS = {'c27': Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v23_20261028/target_compose27'),
       'c21': Path('/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/compose_target')}


def L(p):
    m = trimesh.load(p, force='mesh'); return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def rmean(Rs):
    q = Rot.from_matrix(Rs).as_quat(); q *= np.sign(q @ q[0])[:, None]; return Rot.from_quat(np.median(q, 0)).as_matrix()


def implied(ep, obj, comp='c27', mode='rot', bundle='selected20_final_v6'):
    TD = TDS[comp]; fol = RR / bundle / f'episode_{ep:06d}'
    d = np.load(fol / 'object_poses.npz'); names = [str(n) for n in d['names']]; s = names.index(obj)
    cam = d['camera_from_object'][:, s].copy(); nf = len(cam)
    cand = L(fol / f'meshes/{obj}.obj'); ref = L(TD / f'episode_{ep:06d}/{obj}_mesh_metric.obj'); anc, _ = M._surface_moments(*ref)
    ca, Rb = M._register_object_body(cand, ref, anc, 1_000_003 * ep + 101 * s)
    Mo2 = np.eye(4); Mo2[:3, :3] = Rb; Mo2[:3, 3] = anc - Rb @ ca
    Tc = np.load(TD / f'episode_{ep:06d}/{obj}_trajectory.npz')['T_world_obj']; cmo = np.load(TD / f'episode_{ep:06d}/camera_motion.npz')
    n = min(nf, len(Tc))
    imp = cmo['T_a_from_rectL'] @ np.linalg.inv(cmo['cam0_T_cam'][:n]) @ Tc[:n] @ Mo2
    ang = np.degrees(Rot.from_matrix(np.einsum('fji,fjk->fik', cam[:n, :3, :3], imp[:, :3, :3])).magnitude())
    agree = np.where(ang < 20)[0]; E = np.eye(4)
    if len(agree) >= 10:
        Ec = cam[agree] @ np.linalg.inv(imp[agree]); E[:3, :3] = rmean(Ec[:, :3, :3]); E[:3, 3] = np.median(Ec[:, :3, 3], 0)
    imp = E @ imp
    if mode == 'rot':
        cen = M._surface_moments(*cand)[0]
        for f in range(n):
            pc = cam[f, :3, :3] @ cen + cam[f, :3, 3]; imp[f, :3, 3] = pc - imp[f, :3, :3] @ cen
    if n < nf:  # pad with ours
        imp = np.concatenate([imp, cam[n:]], 0)
    ang = np.degrees(Rot.from_matrix(np.einsum('fji,fjk->fik', cam[:, :3, :3], imp[:, :3, :3])).magnitude())
    world = d['world_from_rgb'] @ imp
    return world, imp, ang, d
