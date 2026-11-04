import sys,json;sys.path.insert(0,'/uniretarget_yam/code')
import numpy as np,mujoco,yam_loop as Y
Y.ONSET_FLOOR=1.4;t=Y.YamTask('/uniretarget_yam/episodes/recon_000037','right',0);m=t.m;d=mujoco.MjData(m);m.geom_margin[:]=.02;mine=np.array([m.body(b).name.startswith('left_') for b in range(m.nbody)]);robot=np.array([m.body(b).name.startswith(('left_','right_')) for b in range(m.nbody)])
d.qpos[:]=t.base_q;mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d);o={}
for i in range(d.ncon):
    c=d.contact[i];u,v=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
    if mine[u]!=mine[v] and not(robot[u] and robot[v]) and c.dist<.012:
        k=f'{m.body(u).name}|{m.body(v).name}|{m.geom(c.geom[0]).name}|{m.geom(c.geom[1]).name}';o[k]=round(100*float(c.dist),2)
print('READY pose near contacts (left arm vs non-robot):',o)
