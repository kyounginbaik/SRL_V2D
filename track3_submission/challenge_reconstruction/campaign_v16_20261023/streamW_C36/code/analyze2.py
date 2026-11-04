import sys, numpy as np
sys.path.insert(0, '/data1/SRL_V2D/challenge_reconstruction/campaign_v16_20261023/streamW_C36/code'); import perframe as P
from scipy.spatial.transform import Rotation as R
corr = np.load('/data1/SRL_V2D/challenge_reconstruction/dexagent_final_recon_20261017/refinement_20261021_ep36/source_corrected/episode_000036/object_poses.npz')
Tw = corr['world_from_object'][:, 0]; Wr = corr['world_from_rgb']
G = P.Gc(); Tc = P.TC @ G
print('Gc', np.round(G, 4), 'rotdeg', np.degrees(R.from_matrix(G[:3,:3]).magnitude()))
A0 = Tw[0] @ np.linalg.inv(Tc[0]); Ref = A0 @ Tc
p, o = P.errors(Ref); print('self-check', p.mean(), o.mean())
cm = np.load(P.CMP + '/camera_motion.npz'); print(list(cm.keys())); C = cm['cam0_T_cam']
# camera chain: our world_from_rgb(f) vs A0 @ cam0_T_cam(f): if consistent, X(f)=inv(A0@C(f)) @ Wr(f) constant
X = np.array([np.linalg.inv(A0 @ C[f]) @ Wr[f] for f in range(len(C))])
print('chain X rot (deg) vs f0:', [round(float(np.degrees(R.from_matrix(X[f,:3,:3] @ X[0,:3,:3].T).magnitude())),2) for f in range(0,407,40)])
print('chain X trans (cm) vs f0:', [round(float(np.linalg.norm(X[f,:3,3]-X[0,:3,3])*100),2) for f in range(0,407,40)])
print('X0 rot deg', np.degrees(R.from_matrix(X[0,:3,:3]).magnitude()), 'trans', X[0,:3,3])
# gravity: world z axis in our world
for f in [0, 50, 100, 200, 300, 350, 380, 406]:
    d = Ref[f,:3,3]-Tw[f,:3,3]; dr = R.from_matrix(Ref[f,:3,:3] @ Tw[f,:3,:3].T)
    print(f'f{f}: ours z {Tw[f,2,3]*100:.2f} ref z {Ref[f,2,3]*100:.2f}  dpos {np.round(d*100,2)}  drot {np.degrees(dr.magnitude()):.1f} rotvec {np.round(np.degrees(dr.as_rotvec()),1)}  ours bodyz-up {np.round(Tw[f,:3,2],2)} ref {np.round(Ref[f,:3,2],2)}')
