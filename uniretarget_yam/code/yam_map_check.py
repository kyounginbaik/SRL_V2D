"""Does the source reference land on YAM correctly? Compare palm frame and fingertips with the native floating hand, and distance to the object."""
import sys,json
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
import yam_lower_only as L,yam_loop as Y
e,side=sys.argv[1:3];b=L.Lower(e,side,0);t=b.t;m=t.m;n=mujoco.MjModel.from_binary_path(e+'/native_model.mjb');nd=mujoco.MjData(n);J=np.load(e+'/plan.npz')['joint_q'][0];nq=n.nq-7*t.poses.shape[1];d=mujoco.MjData(m);C=b.tape(np.zeros(b.size));q=[int(m.jnt_qposadr[j]) for j in t.arm]
pre=f'{side}_sharpa_wave_{side}_';nb=n.body(pre+'hand_C_MC').id;tips={x:n.body(pre+f'{x}_DP').id for x in Y.DIGITS}
for f in (0,15,25,30,34,38,42,50):
    nd.qpos[:nq]=J[f];mujoco.mj_kinematics(n,nd);d.qpos[:]=t.base_q;d.qpos[t.q_arm]=C[f,t.act_arm];d.qpos[t.q_f]=J[f,[int(n.jnt_qposadr[n.joint(pre+m.joint(j).name.split(f'_arm_{side}_',1)[1]).id]) for j in t.fingers]];mujoco.mj_kinematics(m,d)
    pa,ra=t.pose(d);rn=R.from_matrix(nd.xmat[nb].reshape(3,3).copy());o=t.gp_raw[f]
    tip_gap=[100*float(np.linalg.norm(nd.xpos[tips[x]]-d.xpos[t.tips[x][0]])) for x in Y.DIGITS]
    print(json.dumps(dict(t=round(f/Y.FPS,2),palm_gap_cm=round(100*float(np.linalg.norm(nd.xpos[nb]-pa)),1),palm_gap_deg=round(float(np.degrees((rn*ra.inv()).magnitude())),1),tip_gap_cm_max=round(max(tip_gap),1),source_tip_to_object_origin_cm=round(min(100*float(np.linalg.norm(nd.xpos[tips[x]]-o)) for x in Y.DIGITS),1),source_palm_to_object_cm=round(100*float(np.linalg.norm(nd.xpos[nb]-o)),1),table_h_palm_cm=round(100*float(pa@t.up-t.verts.dot(t.up).min()),1))))
