import sys,json
import numpy as np,mujoco,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,"/uniretarget_yam/code")
import yam_loop as Y
Y.ONSET_FLOOR=1.4
t=Y.YamTask("/uniretarget_yam/episodes/episode_000037","right",0);m=t.m;d=mujoco.MjData(m)
tips=[b for b in range(m.nbody) if m.body(b).name.startswith("right_arm_right_") and any(x in m.body(b).name for x in ("DP","distal","tip"))]
print("tip bodies",[m.body(b).name for b in tips][:6],"n",len(tips))
V=t.r0.inv().apply(t.verts-t.o0)
E=Path=None
for name,p in (("fixed","/uniretarget_yam/episodes/episode_000037/yam/flash/fixed/reference.parquet"),("refined","/uniretarget_yam/episodes/episode_000037/yam/flash/refined/reference.parquet"),("adopted(seed23)","/search/seed23/flash/reference.parquet"),("mink_raw","/uniretarget_yam/episodes/episode_000037/yam/flash/mink_raw/reference.parquet")):
    r=pq.read_table(p).to_pylist()[0];q=np.array(r["robot_joint_positions"]);names=r["robot_joint_names"];print("==",name,"raw_motion_file",r["raw_motion_file"],"frames",len(q))
    mp=[]
    for n in names:
        j=(("right" if n[0]=="R" else "left")+"_arm_joint"+n[-1]) if n.startswith(("R_arm_j","L_arm_j")) else (("right" if n.startswith("right") else "left")+"_arm_"+n)
        mp.append(int(m.jnt_qposadr[m.joint(j).id]))
    arm=[names.index(f"R_arm_j{i}") for i in range(1,7)];v=np.abs(np.diff(q[:,arm],axis=0))*20
    print(" arm peak rad/s",round(float(v.max()),2),"frames at the 2.7 cap",int((v.max(axis=1)>2.69).sum()),"of",len(v),"by joint max",v.max(axis=0).round(2).tolist())
    jr=np.array([m.jnt_range[m.joint(f"right_arm_joint{i}").id] for i in range(1,7)]);print(" frames at joint limit",int(((q[:,arm]<=jr[:,0]+.01)|(q[:,arm]>=jr[:,1]-.01)).any(axis=1).sum()))
    rows=[]
    for f in (0,5,10,15,20,25,30,34,38,40,45,50,60,80,100,120,149):
        d.qpos[:]=t.base_q;d.qpos[mp]=q[f];mujoco.mj_kinematics(m,d)
        w=t.gp_raw[f]+t.gr_raw[f].apply(V);tp=np.array([d.xpos[b] for b in tips]);dist=np.linalg.norm(tp[:,None]-w[None],axis=2).min(axis=1)
        rows.append((f,round(100*float(np.linalg.norm(d.xpos[t.palm]-t.gp_raw[f])),1),round(100*float(np.sort(dist)[0]),1),round(100*float(np.sort(dist)[1]),1),round(100*float(np.median(dist)),1)))
    print(" frame, palm-to-spoon cm, nearest fingertip-spoon cm, 2nd, median:");[print("  ",x) for x in rows]
    pf=np.array(r["ee_pose_w"])[:,1,:3];print(" palm travel per frame peak cm",round(100*float(np.linalg.norm(np.diff(pf,axis=0),axis=1).max()),1),"total m",round(float(np.linalg.norm(np.diff(pf,axis=0),axis=1).sum()),2))
print("goal rise cm",{f:round(100*float((t.gp_raw[f]-t.gp_raw[0])@t.up),1) for f in (0,20,30,34,38,40,45,50,60,80)})
