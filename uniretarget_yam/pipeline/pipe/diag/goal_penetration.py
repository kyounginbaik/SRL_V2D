import json,sys,numpy as np,mujoco
for n in map(int,sys.argv[1:]):
    e=f'/uniretarget_yam/episodes/recon_{n:06d}';m=mujoco.MjModel.from_binary_path(e+'/yam/yam_model.mjb');M=json.load(open(e+'/recon_meta.json'));names=[o['name'] for o in M['objects']];z=np.load(e+'/baseline/physics.npz');P=z['object_pose'][:,0];d=mujoco.MjData(m)
    # object_pose columns: pose then goal, one row per tracked object in body order
    Q=z['native_qpos'][:,0];print(n,names,'pose',P.shape,'nq',m.nq)
    free=[j for j in range(m.njnt) if m.jnt_type[j]==0];ids=[int(m.jnt_bodyid[j]) for j in free];nm=[m.body(b).name for b in ids];print(' free bodies',nm)
    adr={m.body(b).name:int(m.jnt_qposadr[j]) for j,b in zip(free,ids)};gx=np.load(e+'/goal_raw.npz');print(' goal keys',list(gx.files))
    N=min(len(P),m.nkey if False else len(P));out=[]
    for f in range(0,len(P),max(1,len(P)//30)):
        d.qpos[:]=m.qpos0
        for k,name in enumerate(names):
            if name in adr:
                # goal of held objects, start pose of frozen ones
                if k<P.shape[1]:d.qpos[adr[name]:adr[name]+3]=P[f,k,7:10];d.qpos[adr[name]+3:adr[name]+7]=P[f,k,10:14][[3,0,1,2]]
        mujoco.mj_forward(m,d);deep={}
        for i in range(d.ncon):
            c=d.contact[i];a,b=m.body(int(m.geom_bodyid[c.geom[0]])).name,m.body(int(m.geom_bodyid[c.geom[1]])).name
            if a in names or b in names:
                if a.startswith(('left_','right_')) or b.startswith(('left_','right_')):continue
                k=tuple(sorted((a,b)));deep[k]=min(deep.get(k,0),float(c.dist))
        out.append((f,{'/'.join(x[:8] for x in k):round(100*v,1) for k,v in deep.items() if v<-.003}))
    print(' deepest cm by frame',[o for o in out if o[1]][:14],flush=True)
