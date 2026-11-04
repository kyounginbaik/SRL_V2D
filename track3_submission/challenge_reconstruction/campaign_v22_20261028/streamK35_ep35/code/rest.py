"""GT-free rest check: spoon pose vs the measured table plane (our stereo support plane). For given frames: min height of mesh
vertices above the plane (mm), tilt of the long axis vs the plane, and the roll (about the long axis) that a rigid mesh resting on
the plane would take (lowest centroid with non-penetration, scanning roll and re-resting) vs the tracked roll.
usage: rest.py BUNDLE_EP_DIR CANON_EP_DIR frames..."""
import sys, numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
B, C = sys.argv[1], sys.argv[2]; frames = [int(x) for x in sys.argv[3:]]
z = np.load(f'{B}/object_poses.npz', allow_pickle=False); T = z['world_from_object'][:, 0]; Wc = z['world_from_rgb']
sp = np.load(f'{C}/support_plane.npz', allow_pickle=False)
import json, glob
mp = glob.glob(f'{B}/meshes/*.obj')[0]; m = trimesh.load(mp, force='mesh'); v = np.asarray(m.vertices); com = m.center_mass if m.is_watertight else v.mean(0)
c = v - v.mean(0); w, U = np.linalg.eigh(c.T @ c / len(c)); long = U[:, 2]
def plane_world(t):
    n = sp['plane_normals_camera'][t]; d = sp['plane_constants_camera'][t]   # n.x + d = 0 in camera
    Rw, tw = Wc[t][:3, :3], Wc[t][:3, 3]; nw = Rw @ n; dw = d - nw @ tw
    if nw @ (Rw @ np.array([0, -1, 0])) < 0: nw, dw = -nw, -dw   # roughly up (camera -y)
    return nw, dw
print('mesh', mp, 'verts', len(v), 'watertight', m.is_watertight)
for t in frames:
    nw, dw = plane_world(t); P = T[t]; vw = v @ P[:3, :3].T + P[:3, 3]; h = vw @ nw + dw
    lw = P[:3, :3] @ long; tilt = np.degrees(np.arcsin(abs(lw @ nw)))
    # roll scan: rotate mesh about its long axis through COM, then drop onto plane (translate along normal) -> COM height
    best = []
    for r in np.arange(-180, 180, 2.0):
        E = Rot.from_rotvec(long * np.radians(r)).as_matrix(); vr = (v - com) @ E.T + com
        vw2 = vr @ P[:3, :3].T + P[:3, 3]; hh = vw2 @ nw + dw; comw = (com @ P[:3, :3].T + P[:3, 3]) @ nw + dw
        best.append((comw - hh.min(), r))
    best.sort(); 
    print(f'f{t:3d} plane_n {nw.round(3)} min_h {h.min()*1000:6.1f}mm  com_h {(com @ P[:3,:3].T + P[:3,3]) @ nw + dw - 0:.3f}  long-axis tilt {tilt:5.1f}deg  '
          f'lowest-COM roll offsets (deg, COM-height mm): ' + ', '.join(f'{r:+.0f}:{hgt*1000:.1f}' for hgt, r in best[:4]) + f' | at 0: {[x for x in best if x[1]==0][0][0]*1000:.1f}mm')
