"""The custom Mink IK (costs, targets, MANO tracks, SavGol) on the YAM + Sharpa model, placed by the spoon into the existing YAM scene."""
import os
import argparse,json,shutil,sys,copy
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco,mink,torch,pyarrow as pa,pyarrow.parquet as pq
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,'rg/scripts/retarget');sys.path.insert(0,'rg/lib')
import ego_recon_to_dexmate_sharpa as D
p=argparse.ArgumentParser();p.add_argument('--allowed',default='');p.add_argument('--plan_sides',default='');p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--arm_speed',type=float,default=2.7);p.add_argument('--iters',type=int,default=100);p.add_argument('--scale',type=float,default=1.2);p.add_argument('--window',type=int,default=11);p.add_argument('--flash',type=Path,default=Path('/scratch2/exaFLOPs/SRL_V2D/uniretarget_yam/episodes/episode_000037/yam/flash'));p.add_argument('--source',default='fixed');p.add_argument('--side',default='right');p.add_argument('--object_index',type=int,default=0);p.add_argument('--plan_out',type=Path);a=p.parse_args()
FL=a.flash;URDF=FL/'yam_sharpa_bimanual.urdf';SD=a.side;PX='R' if SD=='right' else 'L';ik_mod=D.load_ik_module()
src=pq.read_table(FL/a.source/'reference.parquet');row=src.to_pylist()[0];names=row['robot_joint_names'];qfix=np.array(row['robot_joint_positions'],float);ee=np.array(row['ee_pose_w'],float);F=len(qfix)
root=ET.parse(URDF).getroot()
for l in root.findall('link'):
    for c in list(l):
        if c.tag in('visual','collision'):l.remove(c)
ET.SubElement(ET.SubElement(root,'mujoco'),'compiler',{'fusestatic':'false'})
m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));d=mujoco.MjData(m)
adr=np.array([m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)] for n in names]);print('nq',m.nq,'names',len(names),flush=True)
def fk(q,body):
    d.qpos[:]=0;d.qpos[adr]=q;mujoco.mj_kinematics(m,d);b=m.body(body).id;return d.xpos[b].copy(),R.from_matrix(d.xmat[b].reshape(3,3).copy())
warn=[];data=D.load_bundle(a.bundle,0,None,warn);scale=2/(float(data['left_scale'].mean())+float(data['right_scale'].mean()))
from read_mano import MANO
mano=MANO(mano_assets_root='/scratch2/exaFLOPs/latest_recon/mink_handoff/mink_ik_handoff/mano/mano_v1_2',gender='neutral',device=torch.device('cpu'),flat_hand_mean=True,center_idx=None)
tracks={}
for S_ in ('left','right'):
    oc,fp=(D.correct_left_hand_params(data['left_orient_cam'],data['left_finger_pose']) if S_=='left' else (data['right_orient_cam'],data['right_finger_pose']))
    jc,wc=D.mano_camera_fk(mano,S_,data[f'{S_}_betas'],oc,fp,torch.device('cpu'))
    tracks[S_]=D.hand_world_tracks(jc,wc,data['camera_to_world'],data[f'{S_}_wrist_to_world'],scale)
# Spoon anchors the hawor metric world to the YAM world, so hand-to-spoon geometry is kept.
sp=data['object_to_world'][:,:3,3]*scale;q=np.array(row['object_body_position'],float).reshape(F,-1,3)[:,a.object_index]
Rm,_=R.align_vectors(q-q.mean(0),sp-sp.mean(0));tm=q[0]-Rm.apply(sp[0]);fit=np.linalg.norm(Rm.apply(sp)+tm-q,axis=1);print('spoon fit to YAM cm mean',round(100*float(fit.mean()),2),'max',round(100*float(fit.max()),2),flush=True)
pj={S_:(Rm.apply(v[0].reshape(-1,3))+tm).reshape(v[0].shape) for S_,v in tracks.items()};pr={S_:Rm.as_matrix()@v[1] for S_,v in tracks.items()}
# Hand reference from the contact schedule: never follow frames the hand tracker lost, and during a hold keep the hand on the object it holds.
META=json.load(open(a.bundle.parent/'recon_meta.json')) if (a.bundle.parent/'recon_meta.json').exists() else {};HOLDS=META.get('holds',[]);NAMES=row['object_body_names'];OBJP=np.array(row['object_body_position'],float).reshape(F,len(NAMES),3);OBJR=[R.from_quat(np.array(row['object_body_wxyz'],float).reshape(F,len(NAMES),4)[:,k],scalar_first=True) for k in range(len(NAMES))]
def mesh_verts(n):
    v=[]
    for l in open(a.bundle.parent.parent.parent/'objects'/a.bundle.parent.name/f'{n}_visual.obj'):
        if l.startswith('v '):v.append([float(x) for x in l.split()[1:4]])
    return np.array(v)
VERTS={k:mesh_verts(n) for k,n in enumerate(NAMES)}
from scipy.spatial import ConvexHull
HULL={k:ConvexHull(v).equations for k,v in VERTS.items()}
def outer(k,p_):
    """Closest point on the object's outer skin, so a hollow object is never touched from inside; the hull normal and signed distance come with it."""
    e_=HULL[k];s_=e_[:,:3]@p_+e_[:,3];i_=int(s_.argmax());return p_-s_[i_]*e_[i_,:3],e_[i_,:3],float(s_[i_])
def fill(j,r,ok):
    """Bridge frames not marked ok by interpolating positions and rotations between the nearest good frames."""
    good=np.flatnonzero(ok);j=j.copy();r=r.copy()
    if not len(good):return j,r
    J=j.shape[1]
    for f in np.flatnonzero(~ok):
        b=good[good<f];c=good[good>f];p_=b[-1] if len(b) else c[0];q_=c[0] if len(c) else b[-1];w=(f-p_)/(q_-p_) if q_!=p_ else 0.;j[f]=(1-w)*j[p_]+w*j[q_];Rp=R.from_matrix(r[p_].reshape(J,3,3));r[f]=(Rp*R.from_rotvec(w*(Rp.inv()*R.from_matrix(r[q_].reshape(J,3,3))).as_rotvec())).as_matrix()
    return j,r
def blend(j0,r0,j1,r1,w):
    J=j0.shape[1];out_j=j0.copy();out_r=r0.copy()
    for f in np.flatnonzero(w>0):
        out_j[f]=(1-w[f])*j0[f]+w[f]*j1[f];Rp=R.from_matrix(r0[f].reshape(J,3,3));out_r[f]=(Rp*R.from_rotvec(w[f]*(Rp.inv()*R.from_matrix(r1[f].reshape(J,3,3))).as_rotvec())).as_matrix()
    return out_j,out_r
main_npz=np.load(a.bundle/'result.npz');HAND_REP={}
for S_ in ('left','right'):
    v_=main_npz[f'hand_{S_}_is_valid'].astype(bool)[:F];pj[S_],pr[S_]=fill(pj[S_],pr[S_],v_);HAND_REP[S_]=dict(valid=round(float(v_.mean()),2),holds=[])
    for x in [x for x in HOLDS if x['side']==S_]:
        k=int(x['index']);bd=a.bundle if k==a.object_index else a.bundle.parent/f'bundle_{k}'
        if not (bd/'result.npz').exists():HAND_REP[S_]['holds'].append(dict(object=NAMES[k],skipped='no bundle'));continue
        w_=[];dk=D.load_bundle(bd,0,None,w_);sck=2/(float(dk['left_scale'].mean())+float(dk['right_scale'].mean()));oc_,fp_=(D.correct_left_hand_params(dk['left_orient_cam'],dk['left_finger_pose']) if S_=='left' else (dk['right_orient_cam'],dk['right_finger_pose']));jc_,wc_=D.mano_camera_fk(mano,S_,dk[f'{S_}_betas'],oc_,fp_,torch.device('cpu'));jw_,rw_=D.hand_world_tracks(jc_,wc_,dk['camera_to_world'],dk[f'{S_}_wrist_to_world'],sck)
        spk=dk['object_to_world'][:,:3,3]*sck;Z=np.load(bd/'result.npz');Rk,_=R.align_vectors(OBJP[:,k]-OBJP[:,k].mean(0),spk-spk.mean(0));tk=OBJP[0,k]-Rk.apply(spk[0]);jk=(Rk.apply(jw_.reshape(-1,3))+tk).reshape(jw_.shape);rk=Rk.as_matrix()@rw_
        # As the zip's README says: keep the hand-to-object relation of the hand run, applied to our object track.
        jk=jk+Rk.apply(spk-Z['object_hawor_metric'][:F])[:,None];Ro=OBJR[k];J=jk.shape[1];RoM=Ro.as_matrix();jl=np.einsum('fji,fnj->fni',RoM,jk-OBJP[:,k][:,None]);rl=np.einsum('fij,fnjk->fnik',RoM.transpose(0,2,1),rk);dist=np.array([np.linalg.norm(VERTS[k]-jl[f,0],axis=1).min() for f in range(F)]);ok=Z[f'hand_{S_}_is_valid'].astype(bool)[:F]&(dist<.20);s_=int(x['start']);e_=min(F-1,int(x.get('release') or x.get('object_end') or x['end']));win=np.zeros(F,bool);win[s_:e_+1]=True;hold_ok=ok&win
        if not hold_ok.any():HAND_REP[S_]['holds'].append(dict(object=NAMES[k],skipped='no trusted frame'));continue
        jf,rf=fill(jl,rl,hold_ok);wj=np.einsum('fij,fnj->fni',RoM,jf)+OBJP[:,k][:,None];wr=np.einsum('fij,fnjk->fnik',RoM,rf);wgt=np.clip((np.arange(F)-(s_-8))/8.,0,1)*np.clip(((e_+8)-np.arange(F))/8.,0,1);pj[S_],pr[S_]=blend(pj[S_],pr[S_],wj,wr,wgt);HAND_REP[S_]['holds'].append(dict(object=NAMES[k],frames=[s_,e_],trusted=round(float(hold_ok[s_:e_+1].mean()),2)))
print(json.dumps(dict(hand_repair=HAND_REP)),flush=True)
specs=dict(ik_mod.DEFAULT_FRAME_TASK_SPECS);tn=tuple(specs);tself=tuple(n for n in tn if n.startswith(SD+'_'))
if a.plan_out:
    def maker(SD,PX):
        # Scene and robot are only placed relative to each other: pick the shift where this hand path is most reachable.
        armn=[f'{PX}_arm_j{i}' for i in range(1,7)];jid=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in armn];qa=[int(m.jnt_qposadr[j]) for j in jid];da=[int(m.jnt_dofadr[j]) for j in jid];lo=m.jnt_range[jid,0];hi=m.jnt_range[jid,1];bid=m.body(SD+'_hand_C_MC').id;hold_=[x for x in json.load(open(a.bundle.parent/'recon_meta.json')).get('holds',[]) if x['side']==SD];fi=sorted({f_ for x in hold_ for f_ in range(int(x['start']),min(F,int(x.get('release') or x.get('object_end') or x['end'])+1))}) or list(range(F));fr=np.array(fi)[np.linspace(0,len(fi)-1,30).astype(int)];W0=[D.build_frame_targets(ik_mod,(SD+'_hand_C_MC',),pj,pr,int(f),a.scale)[SD+'_hand_C_MC'] for f in fr];jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
        def path(off):
            d.qpos[:]=0;d.qpos[adr]=qfix[0];q=d.qpos[qa].copy();pe=[];re=[];dq=[]
            for tt in W0:
                tp=tt.position+off;trr=R.from_quat(tt.wxyz,scalar_first=True);prev=q.copy()
                for _ in range(25):
                    d.qpos[qa]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);ep_=tp-d.xpos[bid];er_=(trr*R.from_matrix(d.xmat[bid].reshape(3,3).copy()).inv()).as_rotvec();mujoco.mj_jacBody(m,d,jp,jr,bid);J=np.vstack([jp[:,da],.3*jr[:,da]]);e=np.r_[ep_,.3*er_];q=np.clip(q+np.clip(J.T@np.linalg.solve(J@J.T+1e-3*np.eye(6),e),-.2,.2),lo,hi)
                d.qpos[qa]=q;mujoco.mj_kinematics(m,d);pe.append(np.linalg.norm(tp-d.xpos[bid]));re.append((trr*R.from_matrix(d.xmat[bid].reshape(3,3).copy()).inv()).magnitude());dq.append(np.abs(q-prev).max())
            pe=np.array(pe);re=np.array(re);return float(pe.mean()+.5*np.percentile(pe,90)+.03*re.mean()+.02*np.mean(dq)),float(pe.mean()),float(np.percentile(pe,90)),float(np.degrees(re.mean()))
        return path
    paths=[maker(S_,'R' if S_=='right' else 'L') for S_ in (a.plan_sides.split(',') if a.plan_sides else [SD])]
    def path(off):
        rs=[f_(off) for f_ in paths];return tuple(float(x_) for x_ in np.mean(rs,axis=0))
    grid=[(x,y) for x in np.arange(-.25,.251,.05) for y in np.arange(-.15,.301,.05)];grid=[(x,y) for x,y in grid if not a.allowed or [round(float(x),2),round(float(y),2)] in [[u+0.,v+0.] for u,v in json.load(open(a.allowed))['allowed']]];assert grid,'no collision-free placement';res=[(path(np.array([x,y,0.])),x,y) for x,y in grid];base=path(np.zeros(3));best=min(res,key=lambda r:r[0][0]+.03*np.hypot(r[1],r[2]));a.plan_out.write_text(json.dumps(dict(offset=[float(best[1]),float(best[2]),0.],score=best[0],at_zero=base,candidates=[[float(r[1]),float(r[2]),float(r[0][0]+.03*np.hypot(r[1],r[2]))] for r in res])));print(json.dumps(dict(plan_offset=[round(float(best[1]),2),round(float(best[2]),2)],cm=round(100*best[0][1],1),cm90=round(100*best[0][2],1),deg=round(best[0][3],1),at_zero_cm=round(100*base[1],1),at_zero_cm90=round(100*base[2],1),at_zero_deg=round(base[3],1))),flush=True);raise SystemExit(0)
def cs(n):return 2.0 if n.endswith('hand_C_MC') or n.endswith('_fingertip') else 1.5
PALM_ORI=float(os.environ.get('IK_PALM_ORI','0.2'));PALM_POS=float(os.environ.get('IK_PALM_POS','0.2'));TIP_ORI=float(os.environ.get('IK_TIP_ORI','-1'))
def ori(n,s):
    if n.endswith('hand_C_MC'):return PALM_ORI
    return s.orientation_cost if TIP_ORI<0 else TIP_ORI*(s.orientation_cost>0)
tasks={n:mink.FrameTask(frame_name=n,frame_type='body',position_cost=(PALM_POS if n.endswith('hand_C_MC') else s.position_cost)*cs(n),orientation_cost=ori(n,s),lm_damping=1e-3) for n,s in specs.items()}
PJ_BEFORE_DOCK={k_:v_.copy() for k_,v_ in pj.items()};PR_BEFORE_DOCK={k_:v_ for k_,v_ in pr.items()}
# Fingertips reach the object: during each hold they aim at fixed points on its surface, where the human's fingers were.
CONTACT={}
for i_,x in enumerate(HOLDS):
    k=int(x['index']);S_=x['side'];s_=max(int(x['start']),int(x['onset'])-2);e_=min(F-1,int(x.get('release') or x.get('object_end') or x['end']));fr_=np.linspace(s_,e_,min(25,e_-s_+1)).astype(int);ns=[n for n in tn if n.startswith(S_+'_') and 'fingertip' in n];loc={n:[] for n in ns}
    for f in fr_:
        tg=D.build_frame_targets(ik_mod,tuple(ns),pj,pr,int(f),a.scale)
        for n in ns:loc[n].append(OBJR[k][int(f)].inv().apply(tg[n].position-OBJP[int(f),k]))
    c_=VERTS[k].mean(0);CONTACT[i_]={};M_={n:np.median(loc[n],0) for n in ns};dock=np.zeros(3)
    # Dock: a hand the tracker left beside the object moves onto it, as a whole, keeping its shape.
    for _ in range(6):
        t_=np.mean([outer(k,M_[n])[0]-M_[n] for n in ns],0)
        for n in ns:M_[n]=M_[n]+t_
        dock=dock+t_
    wgt_=np.clip((np.arange(F)-(s_-10))/10.,0,1)*np.clip(((e_+8)-np.arange(F))/8.,0,1);pj[S_]=pj[S_]+(wgt_[:,None]*OBJR[k].apply(np.tile(dock,(F,1))))[:,None,:]
    dist_={n:abs(outer(k,M_[n])[2]) for n in ns};use=[n for n in ns if dist_[n]<.08] or sorted(ns,key=dist_.get)[:3]
    for n in use:
        v_,u_,_=outer(k,M_[n]);CONTACT[i_][n]=v_+.007*u_
    x['dock_cm']=round(100*float(np.linalg.norm(dock)),1)
def targets(f):
    tg=D.build_frame_targets(ik_mod,tn,pj,pr,f,a.scale)
    for i_,x in enumerate(HOLDS):
        k=int(x['index']);s_=int(x['start']);on=max(s_,int(x['onset'])-2);e_=min(F-1,int(x.get('release') or x.get('object_end') or x['end']));w=float(np.clip((f-(on-10))/10.,0,1)*np.clip(((e_+8)-f)/8.,0,1))
        if w<=0:continue
        for n,Lc in CONTACT[i_].items():
            p_=OBJP[f,k]+OBJR[k][f].apply(Lc);tg[n]=ik_mod.FrameTarget(position=(1-w)*tg[n].position+w*p_,wxyz=tg[n].wxyz)
    return tg
post=mink.PostureTask(m,cost=1e-2,lm_damping=1e-3)
q0=np.zeros(m.nq);q0[adr]=qfix[0];post.set_target(q0);cfg=mink.Configuration(m,q=q0)
arm=[f'{PX}_arm_j{i}' for i in range(1,7)];both=[f'{x}_arm_j{i}' for x in 'LR' for i in range(1,7)];dt=1e-2;v=a.arm_speed/20/(a.iters*dt);vel=mink.VelocityLimit(m,{n:v for n in both})
hold=[]
limits=[mink.ConfigurationLimit(m),vel]
Q=np.zeros((F,len(names)));Err={n:[] for n in tn};first=True
for f in range(F):
    tg=targets(f)
    for n,t in tg.items():
        tasks[n].set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(R.from_quat(t.wxyz,scalar_first=True).as_matrix()),t.position.copy()))
    best=None;bs=np.inf;lim=[mink.ConfigurationLimit(m)] if first else limits;its=a.iters*(6 if first else 1)
    for _ in range(its):
        u=mink.solve_ik(cfg,[*tasks.values(),post],dt,'daqp',limits=lim,safety_break=False);cfg.integrate_inplace(u,dt)
        s=sum(float(np.linalg.norm(tasks[n].compute_error(cfg)[:3])) for n in tn)
        if s<bs:bs,best=s,cfg.q.copy()
    cfg.update(best);first=False;x=best[adr].copy();x[hold]=qfix[f,hold];Q[f]=x
    for n in tn:Err[n].append(float(np.linalg.norm(tasks[n].compute_error(cfg)[:3])))
    if f%25==0:print('frame',f,'wrist cm',round(100*Err[SD+'_hand_C_MC'][-1],2),'tips cm',round(100*np.mean([Err[n][-1] for n in tself if 'fingertip' in n]),2),flush=True)
if a.window:Q=savgol_filter(Q,a.window,3,axis=0)
for i,n in enumerate(names):
    j=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n);Q[:,i]=np.clip(Q[:,i],*m.jnt_range[j])
Q[:,hold]=qfix[:,hold]
E_=ee.copy();wr,tp=[],[]
for f in range(F):
    for k,b in enumerate(('left_hand_C_MC','right_hand_C_MC')):
        pp,rr=fk(Q[f],b);E_[f,k,:3]=pp;E_[f,k,3:]=rr.as_quat(scalar_first=True)
    tg=targets(f);wr.append(np.linalg.norm(fk(Q[f],SD+'_hand_C_MC')[0]-tg[SD+'_hand_C_MC'].position));tp.append(np.mean([np.linalg.norm(fk(Q[f],n)[0]-tg[n].position) for n in tself if 'fingertip' in n]))
ia=[names.index(n) for n in arm];sp_=np.abs(np.diff(Q[:,ia],axis=0)).max(axis=1)*20
row['robot_joint_positions']=Q.tolist();row['ee_pose_w']=E_.tolist();row['raw_motion_file']=str(a.bundle);row['sequence_id']=a.output.name
a.output.mkdir(parents=True,exist_ok=True);pq.write_table(pa.Table.from_pylist([row],schema=src.schema),a.output/'reference.parquet')
for n in('scene_overrides.json','reference_support.usda'):shutil.copy(FL/a.source/n,a.output/n)
contact_rep=[]
for i_,x in enumerate(HOLDS):
    k=int(x['index']);S_=x['side'];s_=max(int(x['start']),int(x['onset']));e_=min(F-1,int(x.get('release') or x.get('object_end') or x['end']));ns=[n for n in tn if n.startswith(S_+'_') and 'fingertip' in n];fr2=[];mn=[];inside=[]
    for f in range(s_,e_+1,3):
        L=R.inv(OBJR[k][f]).apply(np.array([fk(Q[f],n)[0] for n in ns])-OBJP[f,k]);sd_=np.array([outer(k,l_)[2] for l_ in L]);ds=np.abs(sd_);fr2.append((ds<.02).sum()>=2);mn.append(ds.mean());inside.append((sd_<-.01).mean())
    contact_rep.append(dict(object=NAMES[k],side=S_,frames=[s_,e_],frames_with_2plus_tips_within_2cm=round(float(np.mean(fr2)),2),mean_tip_to_surface_cm=round(100*float(np.mean(mn)),1),tips_inside_skin=round(float(np.mean(inside)),2),contact_fingers=len(CONTACT[i_]),dock_cm=x.get('dock_cm')))
print(json.dumps(dict(reference_contact=contact_rep)),flush=True)
# What the IK was asked for against what it reached, per hand and keypoint, and what the hand data said before contact docking.
_raw=[D.build_frame_targets(ik_mod,tn,PJ_BEFORE_DOCK,PR_BEFORE_DOCK,f,a.scale) for f in range(F)];_goal=[targets(f) for f in range(F)]
np.savez_compressed(a.output/'ik_debug.npz',names=np.array(tn),raw=np.array([[_raw[f][n].position for n in tn] for f in range(F)]),goal=np.array([[_goal[f][n].position for n in tn] for f in range(F)]),robot=np.array([[fk(Q[f],n)[0] for n in tn] for f in range(F)]),raw_wxyz=np.array([[_raw[f][n].wxyz for n in tn] for f in range(F)]),goal_wxyz=np.array([[_goal[f][n].wxyz for n in tn] for f in range(F)]),robot_wxyz=np.array([[fk(Q[f],n)[1].as_quat(scalar_first=True) for n in tn] for f in range(F)]),scale=a.scale,pos_object=np.array(OBJP),quat_object=np.array([OBJR[k].as_quat() for k in range(len(OBJR))]))
rep=dict(reference_contact=contact_rep,hand_repair=HAND_REP,frames=F,wrist_cm=dict(mean=100*float(np.mean(wr)),max=100*float(np.max(wr))),fingertip_cm=dict(mean=100*float(np.mean(tp)),max=100*float(np.max(tp))),arm_peak_rad_s=float(sp_.max()),frames_over_3=int((sp_>3).sum()),frames_at_cap=int((sp_>2.69).sum()));(a.output/'report.json').write_text(json.dumps(rep,indent=1));print(json.dumps(rep),flush=True);print('YAM_IK_DONE')
