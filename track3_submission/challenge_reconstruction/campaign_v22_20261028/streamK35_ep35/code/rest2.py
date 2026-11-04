"""GT-free physical rest check on all spoon episodes. For frames where the spoon lies on the table (gravity frame from our stereo
support plane, table z=0): (a) penetration/float of the lowest vertex, (b) stability: is the COM (vertical projection) inside the
support polygon of the vertices within 1.5 mm of the lowest point, (c) the roll about the long axis (through the COM) that a rigid
body resting on the plane would settle to (lowest COM after dropping onto the plane), as an offset from the tracked roll.
usage: rest2.py BUNDLE EP:f,f,f ..."""
import sys, numpy as np, trimesh
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.transform import Rotation as Rot
B = sys.argv[1]; CAN = '/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation'
def stable(vg, com):
    z0 = vg[:, 2].min(); sup = vg[vg[:, 2] < z0 + 0.0015][:, :2]
    if len(sup) < 3: return False
    try: hull = Delaunay(sup[ConvexHull(sup).vertices]); return bool(hull.find_simplex(com[:2]) >= 0)
    except Exception: return False
for arg in sys.argv[2:]:
    ep, fr = arg.split(':'); ep = int(ep); d = f'{B}/episode_{ep:06d}'
    z = np.load(f'{d}/object_poses.npz', allow_pickle=False); names = list(z['names']); k = names.index('wooden_spoon')
    G = np.load(f'{CAN}/episode_{ep:06d}/support_plane.npz', allow_pickle=False)['gravity_from_canonical']
    W0 = z['world_from_rgb'][0]; ang0 = np.degrees(Rot.from_matrix(W0[:3, :3]).magnitude())
    m = trimesh.load(f'{d}/meshes/wooden_spoon.obj', force='mesh'); v = np.asarray(m.vertices); com = m.center_mass
    c = v - v.mean(0); w, U = np.linalg.eigh(c.T @ c / len(c)); long = U[:, 2]
    if long[np.argmax(np.abs(long))] < 0: long = -long
    out = []
    for t in [int(x) for x in fr.split(',')]:
        P = G @ z['world_from_object'][t, k]; vg = v @ P[:3, :3].T + P[:3, 3]; cg = P[:3, :3] @ com + P[:3, 3]
        tilt = np.degrees(np.arcsin(abs((P[:3, :3] @ long)[2])))
        res = []
        for r in np.arange(-60, 61, 1.0):
            E = Rot.from_rotvec(long * np.radians(r)).as_matrix(); vr = (v - com) @ E.T + com; vgr = vr @ P[:3, :3].T + P[:3, 3]
            res.append((cg[2] - vgr[:, 2].min(), r))
        h0 = [h for h, r in res if r == 0][0]; hb, rb = min(res)
        out.append(f'f{t}: pen {vg[:,2].min()*1000:+.1f}mm tilt {tilt:.1f} stable@0 {stable(vg, cg)} rest-roll {rb:+.0f} (well {(h0-hb)*1000:.1f}mm)')
    print(f'ep{ep} (world vs canon0 {ang0:.1f}deg) | ' + ' | '.join(out), flush=True)
