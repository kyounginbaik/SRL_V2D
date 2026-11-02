"""Where Newton leaves the reference: arm and finger joint error against the reference, with the object's height, by moment."""
import sys,json
import numpy as np
z=np.load(sys.argv[1]+'/poses.npz');q=z['joint_q'];r=z['joint_ref'];n=[str(x) for x in z['joint_names']];i=z['joint_q_ids'];o=z['object_pose'][:,0,0];side=sys.argv[2];pre='R_arm' if side=='right' else 'L_arm';A=[k for k,x in zip(i,n) if x.startswith(pre)];G=[k for k,x in zip(i,n) if x.startswith(side)]
for f in list(range(0,len(q),int(sys.argv[3]))):
    e=np.degrees(q[f,A]-r[min(f+1,len(r)-1),A]);g=np.degrees(q[f,G]-r[min(f+1,len(r)-1),G]);print(json.dumps(dict(t=round(f/20,2),arm_err_deg_max=round(float(np.abs(e).max()),1),finger_err_deg_max=round(float(np.abs(g).max()),1),finger_err_deg_mean=round(float(np.abs(g).mean()),1),obj_cm=round(100*float(np.linalg.norm(o[f,:3]-o[f,7:10])),1),obj_h=round(100*float(o[f,2]-o[0,2]),1),goal_h=round(100*float(o[f,9]-o[0,9]),1))))
