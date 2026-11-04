"""ep30: target_final_v5 poses -> our mesh body frames (scorer registration) -> our physics world (world_C6cut2 frame) by ONE rigid
transform G anchored on the static rack at frame 0 (the sweeper-anchored G0 is invalid here: our physics sweeper f0 is the
half-turn twin of the target's). Writes ep30/target_ourworld.npz (+ lineage json)."""
import sys, json, hashlib, numpy as np, trimesh
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE_ep25_30/code')
from elib import *
from scipy.spatial import cKDTree
I = '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamI_ep30'
OUT = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamE_ep25_30/ep30'
W = f'{I}/world_C6cut2'
names = ['mini_sweeper', 'plastic_dish_rack']
meshes = {'mini_sweeper': f'{I}/staging_C2/components/episode_000030', 'plastic_dish_rack': None}
comp = Path(f'{I}/staging_C2/components/episode_000030')
our = {n: next(comp.rglob(f'meshes/**/{n}.obj')) for n in names}
cz = np.load(f'{I}/compose_C/compose_poses_ourworld.npz'); Cref = cz['world_from_object']  # C world reference (compose)
tr = np.load(f'{W}/episodes/episode_000030/tracking_reference.npz')
X = {}; Tob = {}
for slot, n in enumerate(names):
    X[n], ca, rot, anchor = registration(30, slot, n, our[n]); Tob[n] = target(30, n) @ X[n]
G = Cref[0, 1] @ np.linalg.inv(Tob['plastic_dish_rack'][0])
new = np.stack([np.einsum('ij,njk->nik', G, Tob[n]) for n in names], 1)
S = np.linalg.inv(Cref[0, 0]) @ new[0, 0]
rv = R.from_matrix(S[:3, :3]).as_rotvec(); print('S body twin: deg', np.degrees(np.linalg.norm(rv)), 'axis(body)', rv / np.linalg.norm(rv), 't cm', S[:3, 3] * 100)
print('S axis in world', Cref[0, 0, :3, :3] @ (rv / np.linalg.norm(rv)))
m = trimesh.load(our['mini_sweeper'], force='mesh'); v = np.asarray(m.vertices)
vs = v @ S[:3, :3].T + S[:3, 3]; d = cKDTree(v).query(vs)[0]
print('sweeper self-chamfer under S (mm) mean/p95/max', d.mean() * 1e3, np.percentile(d, 95) * 1e3, d.max() * 1e3, 'extents', m.extents)
for k, n in enumerate(names):
    p, dd = errs(new[:, k], Cref[:, k]); print(n, 'new vs C-world ref per 20f cm', np.round(p[::20], 1), 'deg', np.round(dd[::20]))
print('rack z drift (cm) range', (new[:, 1, 2, 3] - new[0, 1, 2, 3]).min() * 100, (new[:, 1, 2, 3] - new[0, 1, 2, 3]).max() * 100)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
np.savez(f'{OUT}/target_ourworld.npz', world_from_object=new, names=np.array(names), G_ourworld_from_targetbody=G, S_twin_body=S)
json.dump(dict(sources={n: dict(track=f'{TGT}/episode_000030/{n}_trajectory.npz', sha256=sha(f'{TGT}/episode_000030/{n}_trajectory.npz'),
                                 target_mesh_sha256=sha(f'{TGT}/episode_000030/{n}_mesh_metric.obj'), our_mesh=str(our[n]), our_mesh_sha256=sha(our[n]),
                                 X_ourbody_from_targetbody=X[n].tolist()) for n in names},
               transform='T_ourbody = T_target @ X (scorer registration seed 1_000_003*30+101*slot); new = G @ T_ourbody, G = C-world rack ref(f0) @ inv(target rack ourbody(f0)) (static rack anchor; Gt = I since the target rack lands exactly on the physics rack)',
               G=G.tolist(), S_twin_body=S.tolist(),
               per_frame_vs_C_world_ref=dict({n: dict(zip(['cm', 'deg'], [np.round(a, 3).tolist() for a in errs(new[:, k], Cref[:, k])])) for k, n in enumerate(names)})),
          open(f'{OUT}/target_ourworld_lineage.json', 'w'), indent=1)
