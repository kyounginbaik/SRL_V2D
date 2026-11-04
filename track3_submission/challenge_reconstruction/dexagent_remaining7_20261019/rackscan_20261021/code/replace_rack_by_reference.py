"""Fine placement of the scan rack for one episode: rigid world-frame slide (dx, dy) and yaw about world z through the rack
centre (keeps it flat on the table), chosen to minimise the partner object's penetration into the rack collision hulls at its
reference poses (frames from T0 on), with a small preference for staying at the registration placement.
usage: replace_rack_by_reference.py EPISODE T0  -> prints best and writes OUT json"""
import sys, json, numpy as np, trimesh, glob
from scipy.spatial.transform import Rotation as R
W = '/data1/SRL_V2D/challenge_reconstruction/dexagent_remaining7_20261019/rackscan_20261021'
ep, t0 = int(sys.argv[1]), float(sys.argv[2]); n = f'episode_{ep:06d}'
A = f'{W}/collision_recon/{n}/assets'; z = np.load(f'{W}/collision_recon/{n}/object_poses.npz'); T = z['world_from_object']; names = [str(x) for x in z['names']]
ir = names.index('plastic_dish_rack'); other = [x for x in names if x != 'plastic_dish_rack'][0]; io = names.index(other)
Rw, pw = T[0, ir, :3, :3], T[0, ir, :3, 3]
planes = []
for f in sorted(glob.glob(f'{A}/plastic_dish_rack_convex_repaired/part_*.obj')):
    h = trimesh.load(f, force='mesh'); h = h.convex_hull; nrm = h.face_normals; off = np.einsum('ij,ij->i', nrm, h.triangles[:, 0]); planes.append((nrm, off))
om = trimesh.load(f'{A}/{other}_visual.obj', force='mesh'); pts = om.sample(400, seed=0) if hasattr(om, 'sample') else om.vertices
pts = trimesh.sample.sample_surface(om, 400, seed=0)[0]
frames = np.arange(int(t0 * 20), len(T), 2)
Pw = np.concatenate([pts @ T[k, io, :3, :3].T + T[k, io, :3, 3] for k in frames])   # world points
fid = np.repeat(np.arange(len(frames)), len(pts))
vis = trimesh.load(f'{A}/plastic_dish_rack_visual.obj', force='mesh'); c = vis.vertices.mean(0) @ Rw.T + pw   # rack centre in world
def cost(dx, dy, yaw):
    Rz = R.from_euler('z', yaw, degrees=True).as_matrix()
    # world point -> moved-rack body frame: X_body = Rw^T (Rz^T (P - c - d) + c - pw)
    q = ((Pw - c - [dx, dy, 0]) @ Rz + c - pw) @ Rw
    depth = np.zeros(len(q))
    for nrm, off in planes:
        s = (q @ nrm.T - off).max(1); inside = s < 0; depth[inside] = np.maximum(depth[inside], -s[inside])
    per = np.zeros(len(frames)); np.maximum.at(per, fid, depth)
    return float(per.sum()), int((per > .003).sum()), float(per.max())
base = cost(0, 0, 0); res = []
for dx in np.arange(-.03, .0301, .005):
    for dy in np.arange(-.03, .0301, .005):
        for yaw in (-6, -4, -2, 0, 2, 4, 6):
            s, cnt, mx = cost(dx, dy, yaw); res.append((s + 0.02 * (np.hypot(dx, dy) / .01) + 0.002 * abs(yaw), dx, dy, yaw, s, cnt, mx))
res.sort(); b = res[0]
out = dict(episode=ep, partner=other, from_t=t0, base=dict(sum_depth_m=base[0], frames_over_3mm=base[1], max_depth_mm=base[2] * 1000),
           best=dict(dx_m=b[1], dy_m=b[2], yaw_deg=b[3], sum_depth_m=b[4], frames_over_3mm=b[5], max_depth_mm=b[6] * 1000), rack_centre_world=c.tolist(),
           top5=[dict(dx=r[1], dy=r[2], yaw=r[3], frames=r[5], max_mm=r[6] * 1000) for r in res[:5]], n_frames=len(frames))
json.dump(out, open(f'{W}/search/{n}_rack_replacement.json', 'w'), indent=1); print(json.dumps(out, indent=1))
