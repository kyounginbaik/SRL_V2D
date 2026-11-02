import sys,json
import numpy as np
from scipy.spatial.transform import Rotation as R
for f in sys.argv[1:]:
    o=np.load(f+'/poses.npz')['object_pose'];W=o.shape[1];cm=[];deg=[]
    for w in range(W):
        x=o[:,w,0];i=x[0];al=R.from_quat(i[3:7])*R.from_quat(i[10:14]).inv();g=al.apply(x[:,7:10])+i[:3]-al.apply(i[7:10]);r=al*R.from_quat(x[:,10:14]);cm.append(100*np.linalg.norm(x[:,:3]-g,axis=1));deg.append(np.degrees((R.from_quat(x[:,3:7]).inv()*r).magnitude()))
    cm=np.array(cm);deg=np.array(deg);s=json.load(open(f+'/summary.json'));print(json.dumps(dict(run=f.split('/')[-1],mp_sr=s['official_mp_sr'],worlds=W,cm=round(float(cm.mean()),2),deg=round(float(deg.mean()),1),worst_cm=round(float(cm.max()),1),worst_deg=round(float(deg.max()),1),world_mean_cm_max=round(float(cm.mean(1).max()),2),world_mean_deg_max=round(float(deg.mean(1).max()),1),lift_cm=round(float(100*(o[:,:,0,2].max()-o[0,0,0,2])),1),goal_lift_cm=round(float(100*(o[:,0,0,9].max()-o[0,0,0,9])),1))))
