#!/usr/bin/env python3
"""Transfer a verified robot pinch to real board features; validate in physics."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,itertools,multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,mujoco,trimesh
from scipy.spatial.transform import Rotation
from refine_episode25_geometry import W,B,E,H,G,TR,Experiment,servo_rollout,execute_reference,evaluate_files,dump,geom_pen,evaluate,init
from spoon_transfer import DONOR,donor_grasp,hand_joints

def pinch():
    m=mujoco.MjModel.from_xml_path(str(DONOR/'execution/scene_physics.xml'));z=np.load(DONOR/'execution/trajectory.npz');d=mujoco.MjData(m);d.qpos[:]=z['qpos'][56];mujoco.mj_forward(m,d)
    rig=H.HandRig.build(m,'robot_right_','right');bid=m.body('wooden_spoon').id;pts={k:[] for k in ['thumb','index']}
    for c in d.contact:
        b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);other=b2 if b1==bid else b1 if b2==bid else -1
        if other<0:continue
        for finger in pts:
            if '_'+finger+'_' in m.body(other).name:pts[finger].append(c.pos.copy())
    wr=H.euler_to_mat(rig.wrist_euler(d));wp=rig.wrist_pos(d)
    contacts=np.array([np.mean(pts[k],axis=0) for k in ['thumb','index']]);contacts=(contacts-wp)@wr
    _,_,closed,opened,names=donor_grasp(2.8)
    actual=d.qpos[[rig.finger_qpos[k] for k in rig.finger_qpos]].copy()
    narrowing=float(os.environ.get('E25_PINCH_NARROW','0'))
    if narrowing:
        world_dir=wr@(contacts[0]-contacts[1]);world_dir/=np.linalg.norm(world_dir)
        before=np.array([d.site_xpos[rig.fingertip_sites[k]].copy() for k in ['thumb','index']])
        H.finger_ik(m,d,rig,{'thumb':before[0]-narrowing*world_dir,'index':before[1]+narrowing*world_dir},iters=160)
        mujoco.mj_forward(m,d)
        after=np.array([d.site_xpos[rig.fingertip_sites[k]].copy() for k in ['thumb','index']]);contacts+=(after-before)@wr
        new=d.qpos[list(rig.finger_qpos.values())].copy();closed+=new-actual;actual=new
    return contacts,actual,closed,opened

def pinch_pressure(closed,depth):
    if depth==0:return closed.copy()
    m=mujoco.MjModel.from_xml_path(str(DONOR/'execution/scene_physics.xml'));z=np.load(DONOR/'execution/trajectory.npz');d=mujoco.MjData(m);d.qpos[:]=z['qpos'][56];mujoco.mj_forward(m,d)
    rig=H.HandRig.build(m,'robot_right_','right');cols=list(rig.finger_qpos.values());initial=d.qpos[cols].copy()
    p0=d.site_xpos[rig.fingertip_sites['thumb']].copy();p1=d.site_xpos[rig.fingertip_sites['index']].copy();direction=p0-p1;direction/=np.linalg.norm(direction)
    H.finger_ik(m,d,rig,{'thumb':p0-depth*direction,'index':p1+depth*direction},iters=160)
    return closed+d.qpos[cols]-initial

def poses(exp,side,settled,contacts,actual,closed,opened,out):
    m=exp.model;d=mujoco.MjData(m);cols=exp.cols[side];rig=H.HandRig.build(m,f'robot_{side}_',side)
    name='wooden_piece_1' if side=='left' else 'wooden_piece_2';bid=m.body(name).id;oc=exp.objects[name]
    mesh=trimesh.load(Path(exp.spec['source'])/'assets'/f'{name}.obj',process=False)
    vertices=np.array(mesh.vertices);contact=contacts.copy()
    if side=='left':contact=contact@np.diag([1.,-1.,1.])
    center=contact.mean(0);closing=contact[0]-contact[1];closing/=np.linalg.norm(closing)
    y=np.array([0.,1.,0.]);y-=y@closing*closing;y/=np.linalg.norm(y);basis=np.stack([closing,y,np.cross(closing,y)],axis=1)
    objR=Rotation.from_quat(settled[oc+3:oc+7],scalar_first=True).as_matrix();objp=settled[oc:oc+3]
    rows=[];saved=[];anchor=42 if side=='left' else 164
    for sx,sz in itertools.product([-1,1],repeat=2):
        vs=vertices[(vertices[:,1]>.0015)&(vertices[:,0]*sx>.04)&(vertices[:,2]*sz>.015)]
        if len(vs)<10:continue
        peg=(vs.min(0)+vs.max(0))/2;peg[1]=.006
        for yaw in np.arange(0,360,15):
            x=np.array([np.cos(np.deg2rad(yaw)),0.,np.sin(np.deg2rad(yaw))]);ty=np.array([0.,1.,0.]);targetbasis=np.stack([x,ty,np.cross(x,ty)],axis=1)
            for twist in np.arange(0,360,15):
                rot=Rotation.from_rotvec(x*np.deg2rad(twist)).as_matrix()@targetbasis@basis.T
                localp=peg-rot@center;worldR=objR@rot
                q=settled.copy();q[cols[:3]]=objp+objR@localp;q[cols[3:6]]=G.nearest_euler(H.mat_to_euler(worldR),exp.base[anchor,cols[3:6]]);q[cols[6:]]=actual
                d.qpos[:]=q;mujoco.mj_forward(m,d);pen=geom_pen(m,d,rig,bid)
                if pen['table']>.0006 or pen['target']>.004 or pen['other']>.002 or pen['self']>.003:continue
                actual_contacts={finger:[] for finger in ['thumb','index']}
                for c in d.contact:
                    b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);h=b2 if b1==bid else b1 if b2==bid else -1
                    if h<0:continue
                    for finger in actual_contacts:
                        if '_'+finger+'_' in m.body(h).name:actual_contacts[finger].append(c.frame[:3].copy()*(1 if b1==bid else -1))
                opposing=False;dot=1.
                if all(actual_contacts.values()):
                    n=[np.mean(v,axis=0) for v in actual_contacts.values()];dot=float(n[0]@n[1]/np.linalg.norm(n[0])/np.linalg.norm(n[1]));opposing=dot<-.3
                # Contact sites may have slight clearance on this smaller post;
                # retain nearby poses, but prefer actual opposing collision normals.
                score=float(10*pen['target']+np.linalg.norm(q[cols[:3]]-exp.base[anchor,cols[:3]])+.025*(not opposing))
                rows.append({'side':side,'corner':[sx,sz],'yaw':int(yaw),'twist':int(twist),'penetration':pen,'opposing_actual_normals':opposing,'normal_dot':dot,'score':score,'localp':localp.tolist()})
                saved.append((score,localp,rot,peg,rows[-1]))
    # An initial pinch is useless if the subsequent object rotation drives the
    # hand through the table. Check the maintained grasp over the full motion.
    if os.environ.get('E25_HORIZON_GEOMETRY')=='1':
        retained=[]
        for score,lp,lr,peg,row in saved:
            worst=0.;other_worst=0.
            frames=[50,60,80,100,120,135,150,180,220,260,300,330,355] if side=='left' else [190,200,220,260,300,330,355]
            for t in frames:
                qr=exp.target[t].copy();orot=Rotation.from_quat(qr[oc+3:oc+7],scalar_first=True).as_matrix()
                qr[cols[:3]]=qr[oc:oc+3]+orot@lp;qr[cols[3:6]]=G.nearest_euler(H.mat_to_euler(orot@lr),exp.target[t,cols[3:6]]);qr[cols[6:]]=actual
                # The opposite hand is planned independently; its source pose
                # does not constrain this single-hand grasp selection.
                other='right' if side=='left' else 'left';qr[exp.cols[other]]=exp.target[0,exp.cols[other]]
                d.qpos[:]=qr;mujoco.mj_fwdPosition(m,d);pcheck=geom_pen(m,d,rig,bid);worst=max(worst,pcheck['table']);other_worst=max(other_worst,pcheck['other'])
                if worst>.002:break
            row['full_motion_table_penetration_m']=worst;row['full_motion_other_penetration_m']=other_worst
            if worst<.002:retained.append((score+3*other_worst,lp,lr,peg,row))
        saved=retained
    order=sorted(saved,key=lambda r:r[0]);chosen=[]
    for item in order:
        if any(np.linalg.norm(item[1]-old[1])<.025 and Rotation.from_matrix(old[2].T@item[2]).magnitude()<.5 for old in chosen):continue
        chosen.append(item)
        if len(chosen)>=8:break
    dump(out/f'{side}_geometry.json',{'feasible_count':len(rows),'selected':[v[-1] for v in chosen],'all':rows})
    return chosen

def commands(exp,side,grasp,settled,closed,opened,times=None):
    _,localp,localR,_,_=grasp;cols=exp.cols[side];oc=exp.objects['wooden_piece_1' if side=='left' else 'wooden_piece_2'];Q=exp.base[:,cols].copy();N=len(Q)
    begin,pre,arrive,close=times or ((12,26,40,48) if side=='left' else (130,146,166,174))
    sR=Rotation.from_quat(settled[oc+3:oc+7],scalar_first=True);previous=Q[0,3:6]
    smooth=lambda x:np.clip(x,0,1)**2*(3-2*np.clip(x,0,1))
    for t in range(1,N):
        if t<begin:
            if side=='right':Q[t]=Q[0]
            continue
        fade=smooth((t-close)/10.);goal=Rotation.from_quat(exp.base[t,oc+3:oc+7],scalar_first=True)
        R=(sR*Rotation.from_rotvec((sR.inv()*goal).as_rotvec()*fade)).as_matrix();p=(1-fade)*settled[oc:oc+3]+fade*exp.base[t,oc:oc+3]
        r=G.nearest_euler(H.mat_to_euler(R@localR),previous);previous=r;position=p+R@localp
        position[2]+=.07*(1-smooth((t-pre)/(arrive-pre)))
        closing=smooth((t-arrive)/(close-arrive));finger=(1-closing)*opened+closing*closed
        target=np.r_[position,r,finger];approach=smooth((t-begin)/(pre-begin));Q[t]=(1-approach)*exp.base[t,cols]+approach*target
    Q=np.clip(Q,exp.limits[side][:,0],exp.limits[side][:,1]);Q[0]=exp.base[0,cols];return Q

def main(a):
    out=a.output;out.mkdir(parents=True,exist_ok=True);spec=json.loads((W/'refinement/episode_000025/peg_v1/spec.json').read_text());exp=Experiment(spec);dump(out/'spec.json',spec)
    contacts,actual,closed,opened=pinch();settled=np.load(W/'refinement/episode_000025/peg_v1/passive_forecast.npz')['qpos'][-1]
    candidates={side:poses(exp,side,settled,contacts,actual,closed,opened,out) for side in ['left','right']}
    if a.left_indices:candidates['left']=[candidates['left'][i] for i in a.left_indices]
    dump(out/'provenance.json',{'donor':str(DONOR),'frame':56,'source_hand':'right','left_mirror':'mirror local y, same finger joint angles, independently validated exact model convention','HaWoR_role':'original approaches and hand/object timing; executed finger grip is refined using robot-contact evidence','actual_contact_positions_in_right_wrist':contacts,'closed_robot_targets':closed,'opened_robot_targets':opened,'object_targets_unchanged':True})
    if a.left_only:candidates['right']=[None]
    if not all(candidates.values()):dump(out/'failed.json',{'reason':'No collision-feasible transferred pinch','counts':{k:len(v) for k,v in candidates.items()}});return
    tasks=[];combinations=[]
    pressures=a.pressures or ([0.,.0015,.003,.005,.008] if a.pressure_grid else [0.])
    variants={p:pinch_pressure(closed,p) for p in pressures}
    for i,(l,r,pressure) in enumerate(itertools.product(candidates['left'],candidates['right'],pressures)):
        Q=exp.base.copy()
        for side,g in [('left',l),('right',r)]:
            Q[:,exp.cols[side]]=commands(exp,side,g,settled,variants[pressure],opened) if g is not None else Q[0,exp.cols[side]]
        path=out/'proposals'/f'{i:04d}.npz';path.parent.mkdir(exist_ok=True);TR.save_reference(path,exp.model,Q,exp.ref['labels'],exp.fps,exp.ref['segment'],exp.ref['segments'],source='Measured robot pinch transferred to existing rigid board corner posts; physics evaluation required',arms_coupled=False);tasks.append((i,str(path)));combinations.append([l[-1],r[-1] if r is not None else 'right hand parked at exact initial pose',{'inward_fingertip_command_shift_m':pressure}])
    dump(out/'combinations.json',combinations);reports=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(spec,)) as pool:
        for index,r in pool.map(evaluate,tasks):
            row={'index':index,**r};reports.append(row);dump(out/'progress.json',{'count':len(reports),'total':len(tasks),'best':min(reports,key=lambda x:x['cost'])});print(json.dumps(row),flush=True)
    dump(out/'scores.json',reports);best=min(reports,key=lambda x:x['cost']);reference=tasks[best['index']][1]
    execution=execute_reference(spec['scene'],reference,out/'execution',expected_duration_s=(len(exp.base)-1)/exp.fps)
    manifest=json.loads((E/'input_manifest.json').read_text());metrics=evaluate_files(manifest['source'],execution['trajectory'],E/'input_manifest.json',execution['scene'],E/'metric_geometry.json')
    dump(out/'result.json',{'episode':'episode_000025','reference':reference,'best':best,'execution':execution,'metrics':metrics,'learned_policy':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--left-only',action='store_true');p.add_argument('--pressure-grid',action='store_true');p.add_argument('--pressures',type=float,nargs='+');p.add_argument('--left-indices',type=int,nargs='+');main(p.parse_args())
