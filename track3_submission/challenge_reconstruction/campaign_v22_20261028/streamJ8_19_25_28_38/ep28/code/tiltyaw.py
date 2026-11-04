"""per-frame pot rotation error decomposition (tilt of pot axis vs spin about it) achieved run vs world reference. usage: tiltyaw.py RUN_DIR [EPD]"""
import sys, json, numpy as np, trimesh
from scipy.spatial.transform import Rotation as R
run = sys.argv[1]; J = '/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamJ8_19_25_28_38/ep28'
epd = sys.argv[2] if len(sys.argv) > 2 else J + '/world/episode_target/episode_000028'
r = json.load(open(run + '/result.json')); tr = np.load(r['execution']['trajectory']); nm = [str(x) for x in tr['object_names']]
ref = np.load(epd + '/reference_trajectory.npz'); rn = [str(x) for x in ref['object_names']] if 'object_names' in ref else ['plastic_dish_rack', 'white_pot']
v = np.asarray(trimesh.load(J + '/../../target_final_v5/episode_000028/white_pot_mesh_metric.obj', force='mesh').vertices)
i = nm.index('white_pot'); j = rn.index('white_pot'); k = nm.index('plastic_dish_rack')
qa = tr['object_quaternions_wxyz'][:, i]; qr = ref['object_quaternions_wxyz'][:, j]
Ra = R.from_quat(qa[:, [1, 2, 3, 0]]); Rr = R.from_quat(qr[:, [1, 2, 3, 0]])
# pot axis in body frame: smallest-variance principal axis of our body mesh? use the axis that is most vertical at frame 0 of the reference
cands = np.eye(3); up = np.array([0, 0, 1.]); ax = cands[np.argmax(np.abs(Rr[0].as_matrix().T @ up))]
ta = Ra.apply(ax); trf = Rr.apply(ax); tilt_a = np.degrees(np.arccos(np.clip(np.abs(ta[:, 2]), -1, 1))); tilt_r = np.degrees(np.arccos(np.clip(np.abs(trf[:, 2]), -1, 1)))
d = (Rr.inv() * Ra); tot = np.degrees(d.magnitude()); dax = np.degrees(np.arccos(np.clip((ta * trf).sum(1), -1, 1)))
pa = tr['object_positions'][:, i]; pr = ref['object_positions'][:, j]; rk = tr['object_positions'][:, k]
print('axis', ax)
for s in range(0, len(qa), 10):
    e = slice(s, s + 10)
    print(f'f{s:3d} tot {tot[e].mean():5.1f} axisdev {dax[e].mean():5.1f} tilt ach {tilt_a[e].mean():5.1f} ref {tilt_r[e].mean():5.1f} | z ach {pa[e,2].mean()*100:5.1f} ref {pr[e,2].mean()*100:5.1f} dxy {np.linalg.norm(pa[e,:2]-pr[e,:2],axis=1).mean()*100:4.1f} rack {rk[e,2].mean()*100:5.1f}')
