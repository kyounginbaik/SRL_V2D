import sys, numpy as np, mujoco
sys.path.insert(0,'.')
from planner import base_from, XML
d0=np.load(sys.argv[1]); w=0; q=d0['q'][:,w].astype(float); b=d0['b'][:,w].astype(float); o=d0['o'][:,w,0].astype(float)
jl=[str(x) for x in d0['joint_labels']]; qs=d0['joint_q_start']; bl=[str(x) for x in d0['body_labels']][:53]
m=mujoco.MjModel.from_xml_path(XML); d=mujoco.MjData(m); hb=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_hand_C_MC')
from scipy.spatial.transform import Rotation as R
fmap=[(m.jnt_qposadr[j], qs[jl.index('right_sharpa_wave/'+mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j))]) for j in range(6,28)]
pid=m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'right_object_joint')]
PG={i for i in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('right_object_') and 'visual' not in mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i)}
names=[n for n in bl if n.startswith('right_sharpa_wave/right_')]
qq=np.zeros(m.nq); E=[]; P=[]; who={}
for k in range(0,len(q),15):
    X=np.eye(4); X[:3,:3]=R.from_quat(b[k,bl.index('right_sharpa_wave/right_hand_C_MC'),3:7]).as_matrix(); X[:3,3]=b[k,bl.index('right_sharpa_wave/right_hand_C_MC'),:3]
    for a,s in fmap: qq[a]=q[k,s]
    qq[:6]=base_from(m,d,hb,X,qq.copy()); qq[pid:pid+3]=o[k,:3]; qq[pid+3:pid+7]=o[k,[6,3,4,5]]
    d.qpos[:]=qq; mujoco.mj_forward(m,d)
    e={n:np.linalg.norm(d.xpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n.split('/')[1])]-b[k,bl.index(n),:3]) for n in names}
    E.append(max(e.values())); worst=max(e,key=e.get); who[worst]=who.get(worst,0)+1
    for c in range(d.ncon):
        g1,g2=d.contact[c].geom1,d.contact[c].geom2
        if (g1 in PG)!=(g2 in PG) and d.contact[c].dist<-.003:
            hg=g2 if g1 in PG else g1; P.append((mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,hg), round(-d.contact[c].dist*100,2)))
print('fk err cm max/med', round(max(E)*100,2), round(float(np.median(E))*100,2), who)
from collections import Counter; print(Counter(p[0] for p in P).most_common(8)); print(P[:10])
print('q layout check, wrist rotvec free?', [ (jl[i],qs[i],qs[i+1]) for i in range(26,31)])
