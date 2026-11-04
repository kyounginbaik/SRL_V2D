"""Build a YAM episode from the reconstruction zip alone: native scene, goal trace, assets, moving objects and hands.
Conventions: world z is up (from the gravity-aligned camera), the camera looks along +y, so the arms sit on the camera's side."""
import argparse,json,re,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np,mujoco,trimesh,coacd,pyarrow as pa,pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
sys.path.insert(0,str(Path(__file__).parent))
import yam_scene,yam_flash_assets,recon_holds
JOINTS=None
def up_and_heading(zp,hawor):
    W=zp['world_from_rgb'];C=hawor['camera_to_world_transform'];n=len(W);u=np.mean([W[i,:3,:3]@C[int(i*(len(C)-1)/max(1,n-1)),:3,:3].T@np.array([0,0,1.]) for i in range(n)],0);u/=np.linalg.norm(u)
    R1=R.align_vectors([[0,0,1.]],[u])[0];f=R1.apply(W[:,:3,2]).mean(0);yaw=np.arctan2(f[0],f[1]);return u,R.from_euler('z',yaw)*R1
MAX_CLEAR=.03
def decompose(path,out,threshold,max_parts=16):
    m=trimesh.load(path,force='mesh');parts=coacd.run_coacd(coacd.Mesh(m.vertices,m.faces),threshold=threshold,max_convex_hull=max_parts,preprocess_mode='auto');hulls=[]
    for old in out.glob('part_*.obj') if out.is_dir() else ():old.unlink()
    out.mkdir(parents=True,exist_ok=True)
    for i,(v,f) in enumerate(parts):
        h=trimesh.Trimesh(v,f).convex_hull;h.export(out/f'part_{i:03d}.obj');hulls.append(h)
    return m,hulls
def inertia(hulls,area=None,density=450.,lim=(.1,1.2),wall=2.):
    """Mass from convex volume, capped by a 2 mm shell of the surface in kg per m2: household objects are mostly hollow. The reconstruction has no mass, so this is an estimate."""
    vol=sum(h.volume for h in hulls);mass=float(np.clip(min(vol*density,area*wall) if area else vol*density,*lim));com=sum(h.volume*h.center_mass for h in hulls)/vol;I=np.zeros((3,3))
    for h in hulls:
        dd=h.center_mass-com;I+=h.moment_inertia*(1)+h.volume*1.0*(dd@dd*np.eye(3)-np.outer(dd,dd))
    I*=mass/vol;I=.5*(I+I.T);w=np.linalg.eigvalsh(I)
    if w.min()<1e-7:I+=np.eye(3)*(1e-7-w.min())
    return mass,com,I
def main():
    p=argparse.ArgumentParser();p.add_argument('--zip_root',type=Path,required=True);p.add_argument('--episode',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--objects_dir',type=Path,required=True);p.add_argument('--objects_mount',default='/uniretarget_yam/objects');p.add_argument('--joint_names_from',type=Path,required=True);p.add_argument('--threshold',type=float,default=.05);p.add_argument('--offset',default='');a=p.parse_args()
    ep=f'episode_{a.episode:06d}';tg=a.zip_root/'targets'/ep;bd=a.zip_root/'bundles'/ep;hd=a.zip_root/'hands'/ep;zp=np.load(bd/'object_poses.npz');names=[str(x) for x in zp['names']];first=np.load(hd/f'hawor_{names[0]}_result.npz');up,Rw=up_and_heading(zp,first);a.output.mkdir(parents=True,exist_ok=True)
    F=len(np.load(tg/f'{names[0]}_trajectory.npz')['T_world_obj']);K=len(names);T=np.stack([np.load(tg/f'{n}_trajectory.npz')['T_world_obj'] for n in names],1)
    sched=recon_holds.schedule(tg,hd,names,T,up);holds=sched['holds'];frozen=sched['frozen'];mk=sched['primary'];pad=recon_holds.pad_schedule(sched);T=np.concatenate([T[:1]]*pad+[T],0) if pad else T;F=sched['end']+1;T=T[:F];hold=dict(mover=mk,object=names[mk],side=next(x['side'] for x in holds if x['index']==mk),end=sched['end'],full_frames=sched['full_frames'],coverage=sched['coverage'],kinds=sched['kinds'],stages=sched['stages'])
    pos=Rw.apply(T[:,:,:3,3].reshape(-1,3)).reshape(F,K,3);rot=[R.from_matrix(Rw.as_matrix()@T[:,k,:3,:3]) for k in range(K)];c=pos.reshape(-1,3).mean(0);pos=pos-np.r_[c[:2],0.]
    # The round wood table as measured in the recording (diag/table_fit.py); nothing else stands under the objects.
    tfit=json.loads(Path('/scratch2/exaFLOPs/srl_yam_run/source_meshes/table_fits.json').read_text())[f'episode_{a.episode:06d}'];table_xy=(Rw.apply(np.array(tfit['center_world']))-np.r_[c[:2],0.])[:2];TABLE_R=float(tfit['radius'])
    sym={}
    for k_ in sorted({h_['index'] for h_ in holds}):
        y_=recon_holds.symmetry(tg/f'{names[k_]}_mesh_metric.obj')
        if y_:pos[:,k_],rot[k_],d_=recon_holds.despin(pos[:,k_],rot[k_],y_['axis'],y_['center']);sym[names[k_]]=dict(mismatch=y_['mismatch'],**d_)
    raw_goal=dict(pos=pos[:,mk].copy(),quat=rot[mk].as_quat());pos[:,mk],rot[mk],cond=recon_holds.condition(pos[:,mk],rot[mk])
    for k_ in sorted({h_['index'] for h_ in holds}-{mk}):
        raw_goal[f'pos{k_}']=pos[:,k_].copy();raw_goal[f'quat{k_}']=rot[k_].as_quat();pos[:,k_],rot[k_],c_=recon_holds.condition(pos[:,k_],rot[k_]);[h_.update(conditioning=c_) for h_ in holds if h_['index']==k_]
    for k_ in frozen:pos[:,k_],rot[k_],_=recon_holds.condition(pos[:,k_],rot[k_])  # No object is fixed: one no hand holds follows its recorded path, and is scored on it.
    s=mujoco.MjSpec();s.option.timestep=1/400;s.option.gravity=[0,0,-9.81];s.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST;s.option.cone=mujoco.mjtCone.mjCONE_ELLIPTIC;s.option.impratio=10;s.option.iterations=100;s.option.ls_iterations=50
    meshes=[];infos=[];odir=a.objects_dir/f'recon_{a.episode:06d}'
    # The dataset's scanned meshes, put in the body frame of these poses by diag/register_all.py; the reconstruction's mesh only where none exists.
    reg=Path('/scratch2/exaFLOPs/srl_yam_run/source_meshes/registered')/f'episode_{a.episode:06d}'
    collision_src=lambda n:next((p for p in (reg/f'{n}_decomp.obj',reg/f'{n}_collision.obj',reg/f'{n}.obj') if p.is_file()),tg/f'{n}_mesh_metric.obj');visual_src=lambda n:reg/f'{n}.obj' if (reg/f'{n}.obj').is_file() else tg/f'{n}_mesh_metric.obj'
    for k,n in enumerate(names):
        cdir=odir/f'{n}_collision';_,hulls=(decompose(collision_src(n),cdir,min(a.threshold,.02),64) if (reg/f'{n}.obj').is_file() else decompose(collision_src(n),cdir,a.threshold));  # Fine enough to follow the scanned GT mesh, holes and slots included; 0.1-0.5 cm from it on all ten objects.
        mesh=trimesh.load(visual_src(n),force='mesh');mass,com,I=inertia(hulls,mesh.area);meshes.append(mesh);infos.append(dict(name=n,mass=mass,parts=len(hulls),extent_cm=(100*mesh.extents).round(1).tolist()))
        mesh.export(odir/f'{n}_visual.obj');body=s.worldbody.add_body(name=n,pos=pos[0,k],quat=rot[k][0].as_quat()[[3,0,1,2]],mass=mass,ipos=com,fullinertia=[I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]);body.add_freejoint(name=n+'_free')
        for i,h in enumerate(hulls):
            s.add_mesh(name=f'{n}_m{i}',uservert=h.vertices.ravel().tolist(),userface=h.faces.ravel().tolist());body.add_geom(name=f'{n}_g{i}',type=mujoco.mjtGeom.mjGEOM_MESH,meshname=f'{n}_m{i}',friction=[.8,.005,.0001],solref=[.02,1],solimp=[.9,.95,.001,.5,2],condim=4,density=0,rgba=[.8,.5,.3,1])
        L=ET.Element('robot',name=n);lk=ET.SubElement(L,'link',name='object');ine=ET.SubElement(lk,'inertial');ET.SubElement(ine,'origin',xyz=' '.join(f'{v:.6f}' for v in com),rpy='0 0 0');ET.SubElement(ine,'mass',value=f'{mass:.5f}');ET.SubElement(ine,'inertia',ixx=f'{I[0,0]:.4e}',ixy=f'{I[0,1]:.4e}',ixz=f'{I[0,2]:.4e}',iyy=f'{I[1,1]:.4e}',iyz=f'{I[1,2]:.4e}',izz=f'{I[2,2]:.4e}');vis=ET.SubElement(lk,'visual');ET.SubElement(ET.SubElement(vis,'geometry'),'mesh',filename=f'{n}_visual.obj')
        for i in range(len(hulls)):
            co=ET.SubElement(lk,'collision');ET.SubElement(ET.SubElement(co,'geometry'),'mesh',filename=f'{n}_collision/part_{i:03d}.obj')
        ET.ElementTree(L).write(odir/f'{n}.urdf')
    # Table under the objects' lowest resting point.
    low=[];
    for k,m in enumerate(meshes):
        v=rot[k][0].apply(m.vertices)+pos[0,k];low.append(float(v[:,2].min()))
    table=min(low)-.001;xy=table_xy;sup=s.worldbody.add_body(name='support',pos=[xy[0],xy[1],table-.02]);sup.add_geom(name='support_top',type=mujoco.mjtGeom.mjGEOM_CYLINDER,size=[TABLE_R,.02,0],friction=[.8,.005,.0001],rgba=[.45,.22,.09,1],density=0)
    # The reconstruction can start objects inside each other: slide the object nobody moves clear of the mover, so physics does not kick it at frame 0.
    m=s.compile();cleared={}
    for _ in range(4):
        dd=mujoco.MjData(m);mujoco.mj_kinematics(m,dd);mujoco.mj_collision(m,dd);deep={}
        for i in range(dd.ncon):
            con=dd.contact[i];x,y=int(m.geom_bodyid[con.geom[0]]),int(m.geom_bodyid[con.geom[1]]);ni=[m.body(x).name,m.body(y).name]
            if names[mk] in ni and con.dist<-.001:
                o=ni[1] if ni[0]==names[mk] else ni[0]
                if o in names and con.dist<deep.get(o,(0,))[0]:deep[o]=(float(con.dist),con.frame[:3].copy()*(1 if ni[0]==names[mk] else -1))
        if not deep:break
        for o,(dist,nrm) in deep.items():
            k=names.index(o);h=np.r_[nrm[:2],0.];h/=max(np.linalg.norm(h),1e-6);sh=h*(-dist+.002)/max(float(np.linalg.norm(nrm[:2])),.3)
            # Only small overlaps are errors to clear. A big one means the object really sits in this one, so the scene stays as recorded.
            if np.linalg.norm(np.array(cleared.get(o,[0,0,0]))+sh)>MAX_CLEAR:continue
            pos[:,k]+=sh;s.body(o).pos=pos[0,k];cleared[o]=(np.array(cleared.get(o,[0,0,0]))+sh).tolist()
        m=s.compile()
    e=a.output;mujoco.mj_saveModel(m,str(e/'native_model.mjb'),None);(e/'baseline').mkdir(exist_ok=True)
    pose=np.zeros((F-1,1,K,14),np.float32);nq=np.zeros((F-1,1,m.nq),np.float32)
    for k in range(K):
        q=rot[k].as_quat();pose[:,0,k,:3]=pos[:F-1,k];pose[:,0,k,3:7]=q[:F-1];pose[:,0,k,7:10]=pos[:F-1,k];pose[:,0,k,10:14]=q[:F-1];ad=m.jnt_qposadr[m.joint(names[k]+'_free').id];nq[:,0,ad:ad+3]=pos[:F-1,k];nq[:,0,ad+3:ad+7]=q[:F-1][:,[3,0,1,2]]
    np.savez_compressed(e/'baseline/physics.npz',object_pose=pose,native_qpos=nq,native_capture_world_ids=np.array([0]),fps=np.array(20.))
    # Centre the workspace on the object that moves; a static rack must not drag it toward the arm bases.
    cen=pos[:F-1,mk].mean(0)
    if a.offset and Path(a.offset).is_file():cen=cen-np.array(json.loads(Path(a.offset).read_text())['offset'])
    (e/'center.json').write_text(json.dumps(cen.tolist()))
    (e/'yam').mkdir(exist_ok=True);ym,info=yam_scene.build(e,e/'yam');(e/'yam/flash').mkdir(exist_ok=True);urdf=yam_flash_assets.build(e,e/'yam/flash/yam_sharpa_bimanual.urdf')
    # Which object moves, with which hand.
    movers=[]
    for k,n in enumerate(names):
        mv=np.linalg.norm(pos[:,k]-pos[0,k],axis=1);turn=(rot[k]*rot[k][0].inv()).magnitude()
        if k!=mk and mv.max()<.02 and turn.max()<.35:continue
        H=np.load(hd/f'hawor_{n}_result.npz');sc=2/(float(H['hand_left_scale'].mean())+float(H['hand_right_scale'].mean()));O=H['object_to_world_transform'][:,:3,3]*sc;onset=int(np.flatnonzero(mv>.02)[0]) if (mv>.02).any() else int(np.flatnonzero(turn>.35)[0]) if (turn>.35).any() else 0;peak=int(mv.argmax());d={}
        for sd in('left','right'):
            W=H[f'hand_{sd}_wrist_to_world_transform'][:,:3,3]*sc;V=H[f'hand_{sd}_is_valid'].astype(bool);win=np.arange(max(0,onset-5-pad),max(1,min(len(W),peak+1-pad)));win=win[V[win]] if V[win].any() else np.array([],int);d[sd]=float(np.median(np.linalg.norm(W[win]-O[win],axis=1))) if len(win) else 99.
        movers.append(dict(object=n,index=k,move_cm=round(100*float(mv.max()),1),turn_deg=round(float(np.degrees(turn.max())),1),onset=onset,peak=peak,side=hold['side'] if k==mk else min(d,key=d.get),wrist_to_object_cm={x:round(100*v,1) for x,v in d.items()}))
    np.savez_compressed(e/'goal_raw.npz',**raw_goal);other=[x for x in movers if x['index']!=mk];movers=[x for x in movers if x['index']==mk]
    ymeta=dict(untracked=sched['untracked'],symmetry=sym,pad=pad,holds=holds,frozen=[names[k_] for k_ in frozen],cleared_objects=cleared,hold=hold,conditioning=cond,other_movers=other,episode=a.episode,frames=F,fps=20.,up_in_target_world=up.tolist(),table_height=table,objects=infos,movers=movers,base=info['bases']['right']['pos']);(e/'recon_meta.json').write_text(json.dumps(ymeta,indent=1))
    # Source reference: objects on their goals, both arms parked. Later references replace the robot columns.
    src=pq.read_table(a.joint_names_from).to_pylist()[0];jn=src['robot_joint_names'];t=e/'yam/flash/source';t.mkdir(parents=True,exist_ok=True);ad=m;Y=mujoco.MjData(ym);Y.qpos[:]=ym.qpos0;q0=[]
    for n in jn:
        j=(("right" if n[0]=="R" else "left")+"_arm_joint"+n[-1]) if n.startswith(("R_arm_j","L_arm_j")) else (("right" if n.startswith("right") else "left")+"_arm_"+n);i=ym.joint(j).id;lo,hi=ym.jnt_range[i];nm=ym.joint(j).name
        v=dict(zip([f'joint{x}' for x in range(1,7)],(0.,1.536,.945,-.61,0.,0.))).get(j.split('_arm_')[1],None);q0.append(v if v is not None else float(np.clip(.15 if hi>abs(lo)*1.5 else 0.,lo,hi)))
    Qs=np.tile(q0,(F,1));E=np.zeros((F,2,7));E[:,:,3]=1;row=dict(schema_version='motion_v1',motion_kind='single_robot',source_dataset='',raw_motion_file='parked',fps=20.,coord_frame='world',robot_joint_names=jn,robot_root_position=np.zeros((F,3)).tolist(),robot_root_wxyz=np.tile([1.,0,0,0],(F,1)).tolist(),robot_joint_positions=Qs.tolist(),ee_link_names=['left_hand_C_MC','right_hand_C_MC'],ee_pose_w=E.tolist(),hand_sides=['left','right'],object_name='+'.join(names),object_body_names=names,object_mesh_paths=[f'{a.objects_mount}/recon_{a.episode:06d}/{n}_visual.obj' for n in names],object_urdf_paths=[f'{a.objects_mount}/recon_{a.episode:06d}/{n}.urdf' for n in names],object_mesh_radius=[float(np.linalg.norm(x.vertices,axis=1).max()) for x in meshes],object_articulation=np.zeros((F,1)).tolist(),object_body_position=pos.tolist(),object_body_wxyz=np.stack([r.as_quat()[:,[3,0,1,2]] for r in rot],1).tolist(),sequence_id='source',robot_name='yam_sharpa')
    pq.write_table(pa.Table.from_pylist([row],schema=pq.read_table(a.joint_names_from).schema),t/'reference.parquet')
    (t/'reference_support.usda').write_text(f'''#usda 1.0
(
    defaultPrim = "Support"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Support"
{{
    def Cylinder "Table"
    {{
        double radius = {TABLE_R}
        double height = 0.04
        uniform token axis = "Z"
        double3 xformOp:translate = ({xy[0]}, {xy[1]}, {table-.02})
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}
}}
''')
    (t/'scene_overrides.json').write_text(json.dumps(dict(gravity=[0,0,-9.81],source='prepare_recon.py from the reconstruction zip'),indent=1))
    print(json.dumps(ymeta),flush=True)
if __name__=='__main__':main()
