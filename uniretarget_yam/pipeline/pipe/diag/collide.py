import sys,json,glob,collections
sys.path.insert(0,'/uniretarget_yam/code')
import numpy as np,mujoco,yam_loop as Y
Y.ONSET_FLOOR=1.4
def names(m,d,obj):
    o=collections.defaultdict(float);rows=[]
    for i in range(d.ncon):
        c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
        A,B=m.body(a).name,m.body(b).name
        if c.dist<0.0005:
            k=' | '.join(sorted((A,B)));o[k]=min(o[k],float(c.dist))
    return {k:round(100*v,2) for k,v in o.items()}
for label,E,S in [('16','.fail_recon_000016','.old3_recon_000016'),('28','.fail_recon_000028','.old3_recon_000028'),('36','.fail_recon_000036','.old3_recon_000036'),('35','.fail_recon_000035','.old3_recon_000035'),('37','recon_000037','recon_000037'),('38','recon_000038','recon_000038')]:
    e='/uniretarget_yam/episodes/'+E;M=json.load(open(e+'/recon_meta.json'));mv=max(M['movers'],key=lambda x:x['move_cm']);side,k=mv['side'],mv['index'];t=Y.YamTask(e,side,k);t.set_stage('real');t.nominal_physics();m=t.m;d=mujoco.MjData(m)
    mujoco.mj_resetData(m,d);d.qpos[:]=t.base_q;d.qpos[t.qo:t.qo+7]=t.reset_pose;mujoco.mj_forward(m,d)
    other='left' if side=='right' else 'right';po=m.body(f'{other}_arm_{other}_hand_C_MC').id;pa=m.body(f'{side}_arm_{side}_hand_C_MC').id
    print(f'== ep{label} active {side} object {m.body(t.obj).name} onset {t.onset} offset {json.load(open("/uniretarget_yam/episodes/recon_0000"+label+"/place.json"))["offset"] if glob.glob("/uniretarget_yam/episodes/recon_0000"+label+"/place.json") else None} (offset may be newer than this snapshot)')
    print('  frame0 contacts (cm, neg=penetration):',names(m,d,t.obj))
    print('  frame0 object-to-hands cm: active',round(100*float(np.linalg.norm(d.xpos[pa]-d.xpos[t.obj])),1),'parked',round(100*float(np.linalg.norm(d.xpos[po]-d.xpos[t.obj])),1))
    # Parked arm against the object's goal path, no dynamics.
    hit=collections.Counter()
    for f in range(0,t.frames-1,5):
        d.qpos[:]=t.base_q;d.qpos[t.qo:t.qo+3]=t.gp_raw[f];d.qpos[t.qo+3:t.qo+7]=t.gr_raw[f].as_quat()[[3,0,1,2]];mujoco.mj_kinematics(m,d);mujoco.mj_collision(m,d)
        for i in range(d.ncon):
            c=d.contact[i];a,b=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
            if t.obj in (a,b) and c.dist<0 and m.body(a if b==t.obj else b).name.startswith(other):hit[m.body(a if b==t.obj else b).name]+=1
    print('  parked arm hits along goal path (samples of',len(range(0,t.frames-1,5)),'):',dict(hit))
    T=[]
    for f in glob.glob(f'/search/{S}/s*/top.json'):T.append(json.load(open(f))[0])
    for f in glob.glob(f'/search/{S}/s*/pass.json'):T.append(json.load(open(f)))
    if not T:print('  no tape');continue
    b=max(T,key=lambda x:(x['frontier'],x['lift_mm']));C,_=t.compile(b['sample'],t.frames-1);mujoco.mj_resetData(m,d);d.qpos[:]=t.base_q;d.qpos[t.qo:t.qo+7]=t.reset_pose;mujoco.mj_forward(m,d);p0=d.qpos[t.qo:t.qo+3].copy();firsts={};log=[]
    for f in range(min(len(C),t.onset+25)):
        d.ctrl[:]=C[f]
        for _ in range(20):mujoco.mj_step(m,d)
        for i in range(d.ncon):
            c=d.contact[i];a,b_=int(m.geom_bodyid[c.geom[0]]),int(m.geom_bodyid[c.geom[1]])
            if t.obj in(a,b_):
                n=m.body(a if b_==t.obj else b_).name;firsts.setdefault(n,f)
        if f%6==0 or f==t.onset:log.append((f,round(100*float(np.linalg.norm(d.qpos[t.qo:t.qo+3]-p0)),1),round(100*float(np.linalg.norm(d.qpos[t.qo:t.qo+3]-t.gp_raw[f])),1)))
    print('  first contact frame per body:',dict(sorted(firsts.items(),key=lambda x:x[1])))
    print('  (frame, object moved cm, error to goal cm):',log)
