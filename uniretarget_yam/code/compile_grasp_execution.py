"""Compile fitted contact candidates into executable approach/close/follow tapes.

Physics decides: the fit's <1 mm gate is not applied here, because a closing
finger crosses a few millimetres by itself. Only contact sets no finger can
reach (underside faces at a resting pose) and deep fit penetrations are removed.
No per-episode constant: timing comes from the object's first goal motion.
"""
import argparse,json,glob
from pathlib import Path
import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from flash_chord.retargeting.native_inverse_dynamics import NativeScalarPDInverse
from flash_chord.retargeting.contact_wrench import ContactPatch,solve_contact_wrench
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY
from compile_inverse_reference import native_velocity


def minimum_jerk(u):
    u=np.clip(u,0.,1.);return u*u*u*(10-15*u+6*u*u)


def goal_motion_onset(positions,rotations,rest,lift_m=.015,turn_rad=.26):
    moved=np.linalg.norm(positions-positions[rest],axis=1)>lift_m
    turned=(rotations*rotations[rest].inv()).magnitude()>turn_rad
    later=np.flatnonzero((moved|turned)[rest:])
    if not len(later):raise ValueError('object goal never leaves its resting pose')
    return rest+int(later[0])


def reverse_closure(m,data,star,bodies,points,normals,finger_joints,clearance,steps=24,damping=.02):
    """Open by moving each contact point back along its normal: closure reversed.

    Only digits holding a contact move; the others keep their fitted posture.
    Joint limits bind. Returns finger qpos at the pre-closure configuration.
    """
    q=star.copy();data.qpos[:]=q;mujoco.mj_kinematics(m,data)
    local=[data.xmat[b].reshape(3,3).T@(p-data.xpos[b]) for b,p in zip(bodies,points)]
    dofs=np.array([m.jnt_dofadr[j] for j in finger_joints]);adr=np.array([m.jnt_qposadr[j] for j in finger_joints])
    lo=m.jnt_range[finger_joints,0];hi=m.jnt_range[finger_joints,1];jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
    goals=np.asarray(points)-clearance*np.asarray(normals)
    for _ in range(steps):
        data.qpos[:]=q;mujoco.mj_kinematics(m,data);mujoco.mj_comPos(m,data);rows=[];errors=[]
        for b,l,g in zip(bodies,local,goals):
            point=data.xpos[b]+data.xmat[b].reshape(3,3)@l
            mujoco.mj_jac(m,data,jp,jr,point,int(b));rows.append(jp[:,dofs].copy());errors.append(g-point)
        J=np.vstack(rows);e=np.concatenate(errors)
        q[adr]=np.clip(q[adr]+J.T@np.linalg.solve(J@J.T+damping**2*np.eye(len(e)),np.clip(e,-.003,.003)),lo,hi)
    data.qpos[:]=q;mujoco.mj_kinematics(m,data)
    return q[adr],float(max(np.linalg.norm(data.xpos[b]+data.xmat[b].reshape(3,3)@l-g) for b,l,g in zip(bodies,local,goals)))


def path_clearance(m,data,rows,hand_geoms,other_geoms):
    worst=np.inf
    for q in rows:
        data.qpos[:]=q;mujoco.mj_kinematics(m,data)
        worst=min(worst,min(mujoco.mj_geomDistance(m,data,int(h),int(o),.05,None) for h in hand_geoms for o in other_geoms))
    return float(worst)


def balanced_forces(patches,normals,weight,minimum,limit,center):
    limits={i:limit for i in range(len(patches))};minima={i:minimum for i in range(len(patches))}
    solved=solve_contact_wrench(patches,weight,center=center,normal_force_limits=limits,normal_force_minima=minima)
    if solved.feasible:return np.asarray(solved.contact_forces),True
    return minimum*np.asarray(normals),False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--discovery',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=6);p.add_argument('--grip',type=float,action='append')
    p.add_argument('--candidates',help='comma-separated names; default = automatic filtering by fit cost')
    p.add_argument('--arc-height',type=float,default=.06);p.add_argument('--settle-seconds',type=float,default=.05)
    p.add_argument('--closure-clearance',type=float,default=.015);p.add_argument('--closure-seconds',type=float,default=.6)
    p.add_argument('--approach-clearance',type=float,default=.03);p.add_argument('--maximum-penetration',type=float,default=.003)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);grips=a.grip or [3.,8.]
    m=mujoco.MjModel.from_binary_path(str(a.fit/'native_model.mjb'));fit=np.load(a.fit/'fit.npz');report=json.loads((a.fit/'result.json').read_text())
    manifest=json.loads((a.plan/'manifest.json').read_text())
    with np.load(a.plan/'plan.npz') as z:Q=np.asarray(z['joint_q'][0],float);T=np.asarray(z['joint_target'][0],float)
    with np.load(a.discovery/'physics.npz') as d:
        fps=float(d['fps']);jmap=d['native_joint_to_newton_joint'][0];qs=d['newton_joint_q_start'];ds=d['newton_joint_qd_start'];pose=d['object_pose'][:,0,0].copy()
    free=[j for j in range(m.njnt) if m.jnt_type[j]==0][0];obj=int(m.jnt_bodyid[free]);qo=int(m.jnt_qposadr[free])
    up=-m.opt.gravity/np.linalg.norm(m.opt.gravity);mg=float(m.body_mass[obj]*np.linalg.norm(m.opt.gravity))
    side=report['side'];rest=int(report['rest_frame'])
    joints=[j for j in range(m.njnt) if m.body(m.jnt_bodyid[j]).name.startswith(side+'_')]
    slides=[j for j in joints if m.jnt_type[j]==2];ball=[j for j in joints if m.jnt_type[j]==1][0];fingers=[j for j in joints if m.jnt_type[j]==3]
    wrist_body=int(m.jnt_bodyid[ball]);frames=len(Q)
    # Official first-pose alignment, smoothed for planning derivatives only.
    init=pose[0];align=R.from_quat(init[3:7])*R.from_quat(init[10:14]).inv();offset=init[:3]-align.apply(init[7:10])
    gp=align.apply(pose[:,7:10])+offset;gr=align*R.from_quat(pose[:,10:14])
    gp=np.vstack([gp,gp[-1]]);gr=R.concatenate([gr,gr[-1:]])
    onset=goal_motion_onset(gp,gr,rest)
    smooth_p=gaussian_filter1d(gp,1.5,axis=0,mode='nearest')
    q_s=(gr[rest].inv()*gr).as_quat()
    for i in range(1,len(q_s)):
        if q_s[i]@q_s[i-1]<0:q_s[i]=-q_s[i]
    q_s=gaussian_filter1d(q_s,1.5,axis=0,mode='nearest');smooth_r=gr[rest]*R.from_quat(q_s/np.linalg.norm(q_s,axis=1,keepdims=True))
    dtf=lambda f:int(round(f*fps))
    close_end=max(rest+1,onset-dtf(a.settle_seconds));close_start=close_end-dtf(a.closure_seconds);arrive=close_start-dtf(.15);leave=max(1,arrive-dtf(.6))
    # Candidates: automatic filtering by reachability at the resting pose.
    data=mujoco.MjData(m);rows=[];skipped=[]
    for f in sorted(glob.glob(str(a.fit/'candidate_*.npz'))):
        z=np.load(f);row=json.loads(Path(f).with_suffix('.json').read_text());n=z['patch_normals'];pts=z['patch_positions']
        data.qpos[:]=z['qpos'];mujoco.mj_kinematics(m,data)
        verts=[]
        for g in np.flatnonzero(m.geom_bodyid==obj):
            mid=m.geom_dataid[g];s=m.mesh_vertadr[mid];verts.append(data.geom_xpos[g]+m.mesh_vert[s:s+m.mesh_vertnum[mid]]@data.geom_xmat[g].reshape(3,3).T)
        low=(np.vstack(verts)@up).min()
        # patch normal = force direction on the object; +up means a finger below it.
        under=(n@up>.5)&((pts@up-low)<.012)
        if under.any():skipped.append(dict(candidate=Path(f).stem,reason='finger_must_fit_under_resting_object'));continue
        if row['maximum_collision_penetration_m']>a.maximum_penetration:skipped.append(dict(candidate=Path(f).stem,reason='deep_fit_penetration'));continue
        rows.append((row['cost'],Path(f).stem,z))
    rows.sort(key=lambda r:r[0]);rows=rows[:a.count]
    if a.candidates:
        wanted=a.candidates.split(',');rows=[]
        for name in wanted:
            z=np.load(a.fit/(name+'.npz'));rows.append((json.loads((a.fit/(name+'.json')).read_text())['cost'],name,z))
    # Wrist ball gains are the runtime's; recover through checkpoint config.
    action=json.loads(read_checkpoint_metadata(manifest['checkpoint'])[CONFIG_METADATA_KEY])['env']['action']
    inverse=NativeScalarPDInverse(m,ball_drives={ball:(float(action['wrist_kp']),float(action['wrist_tau_max']))})
    names=[Path(a.plan).name+'_unchanged'];qout=[Q.copy()];tout=[T.copy()];info=[]
    qadr=lambda j:int(m.jnt_qposadr[j]);slide_q=[qadr(j) for j in slides];finger_q=[qadr(j) for j in fingers];ball_q=qadr(ball)
    newton_q=lambda j:int(qs[jmap[j]]);newton_d=lambda j:int(ds[jmap[j]])
    def native_row(f,src=Q):
        out=m.qpos0.copy()
        for j in range(m.njnt):
            nj=int(jmap[j])
            if nj<0 or qs[nj]>=Q.shape[1]:continue
            w={0:7,1:4,2:1,3:1}[int(m.jnt_type[j])];v=src[f,qs[nj]:qs[nj]+w]
            out[qadr(j):qadr(j)+w]=v[[3,0,1,2]] if m.jnt_type[j]==1 else v
        return out
    for cost,name,z in rows:
        star=z['qpos'].copy();data.qpos[:]=star;mujoco.mj_kinematics(m,data)
        wrist_p=data.xpos[wrist_body].copy();wrist_r=R.from_matrix(data.xmat[wrist_body].reshape(3,3).copy())
        fixed=wrist_r*R.from_quat(star[ball_q:ball_q+4][[1,2,3,0]]).inv()
        pts=z['patch_positions'];away=wrist_p-pts.mean(0);away/=np.linalg.norm(away)
        src_row=native_row(leave)
        contact_bodies=m.geom_bodyid[z['hand_geom_ids']]
        pre_f,closure_miss=reverse_closure(m,data,star,contact_bodies,pts,z['patch_normals'],fingers,a.closure_clearance)
        obj_p=star[qo:qo+3];obj_r=R.from_quat(star[qo+3:qo+7][[1,2,3,0]])
        for grip in grips:
            nq=Q.copy();nt=T.copy()
            geometry=np.repeat(native_row(0)[None],frames,axis=0)
            for f in range(frames):geometry[f]=native_row(f)
            # After `arrive` the wrist is on the grasp; after close_end it follows the object.
            wrist_ps=np.zeros((frames,3));wrist_rs=[None]*frames;follow=np.zeros(frames)
            for f in range(frames):
                follow[f]=minimum_jerk((f-close_end)/max(1,dtf(.2)))
                delta_p=smooth_p[f]-obj_p;delta_r=smooth_r[f]*obj_r.inv()
                rel=R.from_rotvec(follow[f]*delta_r.as_rotvec())
                wrist_ps[f]=obj_p+follow[f]*delta_p+rel.apply(wrist_p-obj_p)
                wrist_rs[f]=rel*wrist_r
            for f in range(leave,frames-1):
                if f<arrive:
                    u=minimum_jerk((f-leave)/max(1,arrive-leave))
                    data.qpos[:]=src_row;mujoco.mj_kinematics(m,data);p0=data.xpos[wrist_body].copy();r0=R.from_matrix(data.xmat[wrist_body].reshape(3,3).copy())
                    pre=wrist_p+a.approach_clearance*away;pose_p=(1-u)*p0+u*pre+a.arc_height*np.sin(np.pi*u)*up
                    pose_r=r0*R.from_rotvec(u*(r0.inv()*wrist_r).as_rotvec());fing=(1-u)*src_row[finger_q]+u*pre_f
                elif f<close_start:
                    pose_p=wrist_p+a.approach_clearance*away;pose_r=wrist_r;fing=pre_f
                else:
                    u=minimum_jerk((f-close_start)/max(1,close_end-close_start));pose_p=wrist_ps[f]+(1-u)*a.approach_clearance*away;pose_r=wrist_rs[f];fing=(1-u)*pre_f+u*star[finger_q]
                row=geometry[f].copy();row[slide_q]=np.asarray(pose_p)-(wrist_p-star[slide_q])
                row[ball_q:ball_q+4]=(fixed.inv()*pose_r).as_quat()[[3,0,1,2]];row[finger_q]=fing;geometry[f]=row
            # Native velocity/acceleration feedforward with the corrected inverse.
            vel=native_velocity(m,geometry,fps)
            acc=np.gradient(vel,1/fps,axis=0,edge_order=2)
            grip_weight=minimum_jerk((np.arange(frames)-close_start)/max(1,close_end-close_start))
            force_ok=None
            jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
            for f in range(leave,frames-1):
                load=np.zeros(m.nv)
                if grip_weight[f]>0:
                    data.qpos[:]=geometry[f];mujoco.mj_kinematics(m,data);mujoco.mj_comPos(m,data)
                    # Contact points and normals ride rigidly with the wrist-held object.
                    rel=wrist_rs[f]*wrist_r.inv();points=wrist_ps[f]+rel.apply(pts-wrist_p);normals=rel.apply(z['patch_normals'])
                    patches=[]
                    for i,(pp,nn) in enumerate(zip(points,normals)):
                        ax=np.eye(3)[np.argmin(np.abs(nn))];t1=np.cross(nn,ax);t1/=np.linalg.norm(t1);patches.append(ContactPatch(tuple(pp),tuple(nn),tuple(t1),tuple(np.cross(nn,t1)),(.7,.7),i))
                    com=points.mean(0);weight=np.r_[mg*up,np.zeros(3)]
                    forces,force_ok=balanced_forces(patches,normals,weight,grip*mg*grip_weight[f],8*mg*max(1.,grip),com)
                    for geom,pp,ff in zip(z['hand_geom_ids'],points,forces):
                        mujoco.mj_jac(m,data,jp,jr,pp,int(m.geom_bodyid[geom]));load+=jp.T@ff
                out=inverse.targets(geometry[f],vel[f],acc[f],contact_load=load)
                for j,v in out.native_joint_position_targets.items():
                    if j in joints:nt[f,newton_d(j)]=v
                di=newton_d(ball);nt[f,di:di+3]=out.native_ball_rotvec_targets[ball]
                for j in joints:
                    w={1:4,2:1,3:1}[int(m.jnt_type[j])];v=geometry[f][qadr(j):qadr(j)+w]
                    nq[f,newton_q(j):newton_q(j)+w]=v[[1,2,3,0]] if m.jnt_type[j]==1 else v
            others=np.r_[np.flatnonzero(m.geom_bodyid==obj),np.flatnonzero(m.geom_bodyid==0)]
            hand=np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in {int(m.jnt_bodyid[j]) for j in joints} and m.geom_contype[g]+m.geom_conaffinity[g]>0])
            placed=geometry.copy();placed[:,qo:qo+7]=star[qo:qo+7]
            approach=path_clearance(m,data,placed[leave:close_start+1],hand,others);closing=path_clearance(m,data,placed[close_start:close_end+1:2],hand,others)
            names.append(f'{name}_grip{grip:g}');qout.append(nq);tout.append(nt);info.append(dict(name=names[-1],fit_cost=cost,force_balanced=bool(force_ok),approach_clearance_m=approach,closing_clearance_m=closing,closure_miss_m=closure_miss,hand_geoms=z['hand_geom_ids'].tolist()))
    np.savez_compressed(a.output/'plan.npz',joint_q=np.asarray(qout,np.float32),joint_target=np.asarray(tout,np.float32))
    manifest.update(candidates=[dict(name=n,controller='zero residual native targets') for n in names],source_plan=str(a.plan.resolve()),
        grasp_execution=dict(rest=rest,onset=onset,leave=leave,arrive=arrive,close_start=close_start,close_end=close_end,skipped=skipped,info=info),
        no_object_force_or_state_assistance=True,physical_success_verified=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(names=names,rest=rest,onset=onset,leave=leave,arrive=arrive,close_start=close_start,close_end=close_end,skipped=skipped,info=info)),flush=True)

if __name__=='__main__':main()
