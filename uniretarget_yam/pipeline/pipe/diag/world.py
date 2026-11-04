import sys;sys.path.insert(0,'/uniretarget_yam/code')
import numpy as np,mujoco,yam_loop as Y,yam_refine as F,pyarrow.parquet as pq
Y.ONSET_FLOOR=1.4;E='/uniretarget_yam/episodes/recon_000037';t=Y.YamTask(E,'right',0);m=t.m;print('world geoms',[(m.geom(g).name,int(m.geom_type[g]),np.round(m.geom_pos[g],2).tolist(),np.round(m.geom_size[g],2).tolist()) for g in np.flatnonzero(m.geom_bodyid==0)][:12]);r=pq.read_table(E+'/yam/flash/mink/reference_both_raw.parquet').to_pylist()[0];q=np.array(r['robot_joint_positions']);mp=F.mapping(m,r['robot_joint_names']);d=mujoco.MjData(m);m.geom_margin[:]=.02
pid=m.body('left_arm_left_hand_C_MC').id;tb=[g for g in range(m.ngeom) if m.body(int(m.geom_bodyid[g])).name=='support'];print('support top z',[float(m.body('support').pos[2]+m.geom_size[g,1]) for g in tb if m.geom_type[g]==5]);base=m.body('left_arm_base_link').id
for f in (0,50,100,150):
    d.qpos[:]=t.base_q;d.qpos[mp]=q[f];mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d);o={}
    for i in range(d.ncon):
        c=d.contact[i];u,v=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
        if 0 in(u,v) and m.body(u if v==0 else v).name.startswith('left_'):k=f'{m.body(u if v==0 else v).name}|{m.geom(c.geom[0] if u==0 else c.geom[1]).name}';o[k]=min(o.get(k,9),round(100*float(c.dist),1))
    print(f,'left palm',np.round(d.xpos[pid],2).tolist(),'base',np.round(d.xpos[base],2).tolist(),'world contacts',dict(list(o.items())[:6]))
d.qpos[:]=t.base_q;mujoco.mj_kinematics(m,d);print('READY palm',np.round(d.xpos[pid],2).tolist())
