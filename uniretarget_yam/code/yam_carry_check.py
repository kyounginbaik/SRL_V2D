import sys,json,numpy as np,mujoco
from pathlib import Path
D=Path(sys.argv[1]);run=Path(sys.argv[2]);m=mujoco.MjModel.from_binary_path(str(D/'yam/yam_model.mjb'));d=mujoco.MjData(m);p=json.loads((D/'yam/yam_placement.json').read_text());up=np.array(p['up']);tab=p['table_height']
z=np.load(run/'physics.npz');Q=z['native_qpos'][:,0];goal=np.load(D/'baseline/physics.npz')['object_pose'][:,0,0];obj=[b for b in range(m.nbody) if m.body(b).name==p['objects'][0]][0];qa=m.jnt_qposadr[m.body_jntadr[obj]]
print('table',round(tab,3),'objects',p['objects'])
rows=[]
for f in range(len(Q)):
    d.qpos[:]=Q[f];mujoco.mj_forward(m,d);hand=table=0;fing=set()
    for c in d.contact[:d.ncon]:
        a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
        if obj not in(a,b):continue
        o=b if a==obj else a;n=m.body(o).name
        if n.startswith('right_'):hand+=1;fing.add(n.replace('right_arm_right_','').split('_')[0])
        elif n.startswith('left_'):fing.add('LEFT')
        else:table+=1
    rows.append((float(Q[f][qa:qa+3]@up-tab),float(goal[min(f,len(goal)-1),7:10]@up-tab),hand,table,len(fing)))
R=np.array(rows)
print('object height above table cm: start %.1f max %.1f end %.1f | goal start %.1f max %.1f end %.1f'%(100*R[0,0],100*R[:,0].max(),100*R[-1,0],100*R[0,1],100*R[:,1].max(),100*R[-1,1]))
print('frames',len(R),'hand contact',int((R[:,2]>0).sum()),'table contact',int((R[:,3]>0).sum()),'hand only (carried)',int(((R[:,2]>0)&(R[:,3]==0)).sum()),'table only',int(((R[:,2]==0)&(R[:,3]>0)).sum()),'neither',int(((R[:,2]==0)&(R[:,3]==0)).sum()))
for f in range(0,len(R),15):print(f,'h %.1f goal %.1f hand %d table %d digits %d'%(100*R[f,0],100*R[f,1],R[f,2],R[f,3],R[f,4]))
