import numpy as np, json, sys
sys.path.insert(0, 'compose_v2d')
from scipy.spatial.transform import Rotation as Rot
tag = sys.argv[1] if len(sys.argv) > 1 else '_grasp'
z = np.load(f'outputs/compose_ep18_mini_sweeper/track{tag}/poses.npz'); r = json.load(open(f'outputs/compose_ep18_mini_sweeper/track{tag}/results.json'))['results']
wd = np.load('outputs/compose_ep18_mini_sweeper/wilor_hands.npz'); fr = {int(f): i for i, f in enumerate(wd['frames'])}
E = z['E']; meta = json.load(open('outputs/compose_ep18_mini_sweeper/meta.json')); R_aL = np.array(meta['T_a_from_rectL'])[:3, :3]
def hand_abs(f):
    return E[f, :3, :3] @ R_aL.T @ Rot.from_rotvec(wd['hand_pose'][fr[f], :3]).as_matrix()
err = {k: dict(zip(r[k]['frames'], r[k]['ori_frames'])) for k in ('compose_grasp', 'icp_track_6dof', 'compose_meshicp') if k in r}
print(' f   npts contact  px   dHand(deg/step)  handCenter(px)    resid_it(mm)  ori_err grasp / icp / meshicp')
for f in range(20, 140, 5):
    dh = np.degrees(np.linalg.norm(Rot.from_matrix(hand_abs(f) @ hand_abs(f - 1).T).as_rotvec())) if f in fr and f - 1 in fr else float('nan')
    c = wd['joints_2d'][fr[f]].mean(0) if f in fr else (np.nan, np.nan)
    ri = z['resid_it'][f] if 'resid_it' in z.files else np.nan
    e = [err[k].get(f, np.nan) for k in ('compose_grasp', 'icp_track_6dof', 'compose_meshicp')]
    print(f'{f:3d} {int(z["n_pts"][f]):6d}    {int(z["contact"][f])} {int(z["contact_px"][f]):6d}      {dh:5.1f}        ({c[0]:5.0f},{c[1]:5.0f})      {ri:6.2f}       {e[0]:5.0f} / {e[1]:5.0f} / {e[2]:5.0f}')
