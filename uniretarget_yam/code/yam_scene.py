"""Build a YAM + Sharpa scene around one episode's objects and table, with vendor meshes and gains."""
import os
import argparse,json,re,tempfile
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation as R
URDF=str(Path(__file__).parent/'yam_assets/yam_sharpa_%s_nogripper.urdf')  # yam_nogripper.py: hand on the wrist, no parallel gripper.
BASE_XY={'left':(-.31,.26),'right':(-.31,-.26)};BASE_ABOVE_TABLE=.06546;READY=dict(zip([f'joint{i}' for i in range(1,7)],(0.,1.536,.945,-.61,0.,0.)))
ARM_GAIN={1:(80.,6.,20.),2:(80.,6.,20.),3:(80.,6.,20.),4:(30.,2.,10.),5:(30.,2.,10.),6:(30.,2.,10.)};ARM_SPEED=3.;ARM_ARMATURE,ARM_FRICTION=.02,.05
FINGER=[(r'_thumb_CMC_',3.3,.0032,.132),(r'_pinky_CMC',.5285,.00012,.012),(r'_MCP_',1.864,.00265,.07456),(r'_(IP|PIP)$',.638,.0006,.01276),(r'_DIP$',.18937,.00042,.00378738)];FINGER_KP,FINGER_KD=2.,.05
def arm_spec(side):
    # Keep palm and fingertip bodies, and the vendor's visual meshes, through URDF import.
    text=Path(URDF%side).read_text();text=re.sub(r'(<robot[^>]*>)',r'\1<mujoco><compiler discardvisual="false" fusestatic="false" balanceinertia="true"/></mujoco>',text,count=1)
    f=tempfile.NamedTemporaryFile('w',suffix='.urdf',delete=False);f.write(text);f.close();return mujoco.MjSpec.from_file(f.name)
def placement(native,poses):
    up=-native.opt.gravity/np.linalg.norm(native.opt.gravity);fx=np.cross([0,1,0],up);fx/=np.linalg.norm(fx);fy=np.cross(up,fx);M=np.c_[fy,-fx,up];c=poses[:,:,:3].reshape(-1,3).mean(0)
    return up,M,c
def build(episode,output):
    e=Path(episode);n=mujoco.MjModel.from_binary_path(str(e/'native_model.mjb'));z=np.load(e/'baseline/physics.npz');poses=z['object_pose'][:,0];q0=z['native_qpos'][0,0];up,M,c=placement(n,poses)
    if (e/'center.json').is_file():c=np.array(json.loads((e/'center.json').read_text()))  # Workspace centred on the object being moved.
    s=mujoco.MjSpec();s.option.timestep=1/400;s.option.gravity=n.opt.gravity;s.option.integrator=n.opt.integrator;s.option.cone=n.opt.cone;s.option.impratio=n.opt.impratio;s.option.solver=n.opt.solver;s.option.iterations=n.opt.iterations;s.option.ls_iterations=n.opt.ls_iterations;s.option.tolerance=n.opt.tolerance;s.visual.global_.offwidth=1920;s.visual.global_.offheight=1080
    d=mujoco.MjData(n);d.qpos[:]=q0;mujoco.mj_kinematics(n,d);keep=[];mesh_ids={};robot=lambda name:name.startswith(('left_','right_'))
    def copy_geom(parent,g,prefix):
        kind=int(n.geom_type[g]);kw=dict(type=mujoco.mjtGeom(kind),pos=n.geom_pos[g],quat=n.geom_quat[g],size=n.geom_size[g],friction=n.geom_friction[g],solref=n.geom_solref[g],solimp=n.geom_solimp[g],condim=int(n.geom_condim[g]),rgba=n.geom_rgba[g],margin=float(n.geom_margin[g]),gap=float(n.geom_gap[g]),density=0)
        g2=parent.add_geom(name=f'{prefix}_g{g}',**kw)
        if kind==mujoco.mjtGeom.mjGEOM_MESH:
            mid=int(n.geom_dataid[g]);name=f'm{mid}'
            if mid not in mesh_ids:
                a,b=n.mesh_vertadr[mid],n.mesh_vertnum[mid];fa,fb=n.mesh_faceadr[mid],n.mesh_facenum[mid];s.add_mesh(name=name,uservert=n.mesh_vert[a:a+b].ravel().tolist(),userface=n.mesh_face[fa:fa+fb].ravel().tolist());mesh_ids[mid]=name
            g2.meshname=mesh_ids[mid]
    # Static geometry and free objects come from the episode's exported model unchanged.
    for g in range(n.ngeom):
        b=int(n.geom_bodyid[g])
        if b==0:copy_geom(s.worldbody,g,'world')
    names={}
    for b in range(1,n.nbody):
        nm=n.body(b).name
        if robot(nm):continue
        if n.body_jntnum[b]==0:
            bb=s.worldbody.add_body(name=nm,pos=n.body_pos[b],quat=n.body_quat[b])
        else:
            bb=s.worldbody.add_body(name=nm,pos=d.xpos[b],quat=d.xquat[b],mass=float(n.body_mass[b]),ipos=n.body_ipos[b],iquat=n.body_iquat[b],inertia=n.body_inertia[b]);bb.add_freejoint(name=nm+'_free');names[nm]=b
        for g in np.flatnonzero(n.geom_bodyid==b):copy_geom(bb,int(g),nm)
    floor=float((poses[:,:,:3].reshape(-1,3)@up).min())-.03
    tabletop=[]
    for g in range(n.ngeom):
        if n.body(int(n.geom_bodyid[g])).name=='support' and n.geom_type[g]==mujoco.mjtGeom.mjGEOM_CYLINDER:tabletop.append(float(d.geom_xpos[g]@up+n.geom_size[g,1]))
    table=max(tabletop) if tabletop else floor
    q=R.from_matrix(M).as_quat()[[3,0,1,2]]
    bases={}
    for side in ('left','right'):
        p=c+M@np.array([BASE_XY[side][0]+float(os.environ.get('YAM_BASE_X','0')),BASE_XY[side][1],0.]);p=p-(p@up)*up+(table+BASE_ABOVE_TABLE+float(os.environ.get('YAM_BASE_UP','0')))*up;q=R.from_matrix(M).as_quat()[[3,0,1,2]];bases[side]=dict(pos=p.tolist(),quat=q.tolist());fr=s.worldbody.add_frame(pos=p,quat=q);arm=arm_spec(side);root=arm.body('base_link') if arm.body('base_link') is not None else arm.worldbody.first_body()
        fr.attach_body(root,f'{side}_arm_','')
    # The base is welded to the world, so MuJoCo's parent filter does not cover the first joint.
    for side in ('left','right'):s.add_exclude(bodyname1=f'{side}_arm_base_link',bodyname2=f'{side}_arm_link_1')
    for j in s.joints:
        nm=j.name
        if 'free' in nm:continue
        if nm.split('_')[-1].startswith('joint') or re.search(r'_arm_joint[1-6]$',nm):
            i=int(nm[-1]);kp,kd,eff=ARM_GAIN[i];j.armature=ARM_ARMATURE;j.frictionloss=ARM_FRICTION;a=s.add_actuator(name='pos_'+nm,target=nm,trntype=mujoco.mjtTrn.mjTRN_JOINT,gaintype=mujoco.mjtGain.mjGAIN_FIXED,biastype=mujoco.mjtBias.mjBIAS_AFFINE);a.gainprm[0]=kp;a.biasprm[0:3]=[0,-kp,-kd];a.forcelimited=1;a.forcerange=[-eff,eff]
        else:
            for pat,eff,arm_,fr_ in FINGER:
                if re.search(pat,nm):
                    j.armature=arm_;j.frictionloss=fr_;a=s.add_actuator(name='pos_'+nm,target=nm,trntype=mujoco.mjtTrn.mjTRN_JOINT,gaintype=mujoco.mjtGain.mjGAIN_FIXED,biastype=mujoco.mjtBias.mjBIAS_AFFINE);a.gainprm[0]=FINGER_KP;a.biasprm[0:3]=[0,-FINGER_KP,-FINGER_KD];a.forcelimited=1;a.forcerange=[-eff,eff];break
    for b in s.bodies:
        if b.name.startswith(('left_','right_')):b.gravcomp=1.
    m=s.compile();mujoco.mj_saveModel(m,str(output/'yam_model.mjb'),None);visual_model(s,e,names,output);m=mujoco.MjModel.from_binary_path(str(output/'yam_model.mjb'));placement_out=dict(bases=bases,up=up.tolist(),center=c.tolist(),table_height=table,objects=list(names),arm_speed_limit=ARM_SPEED);(output/'yam_placement.json').write_text(json.dumps(placement_out,indent=1)+'\n');return m,placement_out
def visual_model(s,e,names,output):
    """A copy for videos only: the real object mesh shown, the convex parts hidden. Physics and the search never load it."""
    import trimesh
    for n in names:
        p=e.parent.parent/'objects'/e.name/f'{n}_visual.obj'
        if not p.is_file():continue
        t=trimesh.load(p,force='mesh');b=s.body(n)
        for g in b.geoms:g.group=3
        s.add_mesh(name=f'{n}_visual_mesh',uservert=t.vertices.ravel().tolist(),userface=t.faces.ravel().tolist(),inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL);b.add_geom(name=f'{n}_visual_geom',type=mujoco.mjtGeom.mjGEOM_MESH,meshname=f'{n}_visual_mesh',contype=0,conaffinity=0,group=0,density=0,mass=0)
    mujoco.mj_saveModel(s.compile(),str(output/'yam_model_visual.mjb'),None)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('episode');a=p.parse_args();e=Path(a.episode);out=e/'yam';out.mkdir(exist_ok=True);m,info=build(e,out);print(json.dumps(dict(nq=m.nq,nv=m.nv,nu=m.nu,nbody=m.nbody,ngeom=m.ngeom,nmesh=m.nmesh,table=round(info['table_height'],3),base=info['bases']['right']['pos'])))
    d=mujoco.MjData(m)
    for j in range(m.njnt):
        nm=m.joint(j).name;k=re.search(r'_arm_joint([1-6])$',nm)
        if k:d.qpos[m.jnt_qposadr[j]]=list(READY.values())[int(k.group(1))-1]
    for a_ in range(m.nu):d.ctrl[a_]=d.qpos[m.jnt_qposadr[m.actuator_trnid[a_,0]]]
    import time;t=time.monotonic();mujoco.mj_forward(m,d);q0=d.qpos.copy()
    for _ in range(400):mujoco.mj_step(m,d)
    arm=[m.jnt_qposadr[j] for j in range(m.njnt) if 'arm_joint' in m.joint(j).name];palm=m.body('right_arm_right_hand_C_MC').id if mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'right_arm_right_hand_C_MC')>=0 else -1
    print(json.dumps(dict(sim_1s_wall=round(time.monotonic()-t,2),arm_drift_rad=round(float(np.abs(d.qpos[arm]-q0[arm]).max()),3),ncon=int(d.ncon),self_or_table=[(m.body(int(m.geom_bodyid[c.geom[0]])).name[-16:],m.body(int(m.geom_bodyid[c.geom[1]])).name[-16:]) for c in d.contact[:d.ncon]][:6],palm_z_above_table=round(float(d.xpos[palm]@np.array(info['up'])-info['table_height']),3) if palm>=0 else None,visual_geoms=int((m.geom_contype==0).sum()))))
