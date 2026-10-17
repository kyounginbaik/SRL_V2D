"""Plan contact-normal finger opening and clear robot withdrawal after support.

This is a robot-command planner. It never alters object states, targets, times,
physical parameters or actuator limits. Callers must execute and verify the
resulting full trajectory; a supported anchor alone does not prove good release.
"""
from pathlib import Path
import hashlib,sys
import mujoco,numpy as np
from scipy.optimize import least_squares

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H,tracking as TR


def smooth(x):
    x=np.clip(x,0.,1.);return x**3*(10.-15.*x+6.*x*x)


def plan_supported_release(scene,trajectory,reference,*,side,object_name,
        start_s,open_complete_s,clear_s,opening_m=.008,
        withdrawal_horizontal_m=.12,withdrawal_up_m=.08,
        minimum_support_force_n=.1,maximum_joint_increment_rad=.35):
    """Return (Q, metadata) in the source reference's exact model/time convention.

    Pass scene and trajectory from one independently recorded physical run.
    start_s must be an existing20Hz source frame. The physical prefix is
    integrated exactly to obtain true contact forces; IK uses a separate copy.
    Only the selected hand's commands after start_s change. The caller should
    save Q with the original TR metadata, run full physics, assert the recorded
    substep prefix agrees exactly, and audit both hand limits/contact retention.
    """
    if side not in ('left','right'):raise ValueError(side)
    if not 0<=start_s<open_complete_s<clear_s:raise ValueError('Ordered release times required')
    m=mujoco.MjModel.from_xml_path(str(scene));ref=TR.load_reference(reference,m);Q=ref['qpos'].copy();fps=ref['fps']
    k=round(start_s*fps)
    if abs(k/fps-start_s)>1e-9:raise ValueError('Release must start on an original reference frame')
    with np.load(trajectory,allow_pickle=False) as z:
        if str(z['scene_sha256'])!=hashlib.sha256(Path(scene).read_bytes()).hexdigest():raise ValueError('Recorded physical scene differs')
        initial=z['initial_state'];state_spec=int(z['state_spec']);ctrl=z['substep_ctrl'];old_q=z['substep_qpos']
    count=round(start_s/m.opt.timestep);d=mujoco.MjData(m);mujoco.mj_setState(m,d,initial,state_spec)
    for action in ctrl[:count]:d.ctrl[:]=action;mujoco.mj_step(m,d)
    prefix_error=float(abs(d.qpos-old_q[count-1]).max()) if count else 0.
    if prefix_error!=0.:raise ValueError(f'Recorded prefix replay differs: {prefix_error}')
    rig=H.HandRig.build(m,f'robot_{side}_',side);wc=np.array([rig.wrist_qpos[n] for n in ['x','y','z','rx','ry','rz']]);fc=np.array(list(rig.finger_qpos.values()));aids=np.array(list(rig.finger_act.values()));limits=m.jnt_range[m.actuator_trnid[aids,0]];ob=m.body(object_name).id
    normals={};forces={};support_forces={};force=np.zeros(6)
    for i,c in enumerate(d.contact):
        b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);other=b2 if b1==ob else b1 if b2==ob else -1
        if other<0:continue
        mujoco.mj_contactForce(m,d,i,force);fn=max(0.,float(force[0]));name=m.body(other).name
        if not name.startswith('robot_'):support_forces[name]=support_forces.get(name,0.)+fn
        if other not in rig.body_ids or fn<.1:continue
        f=next((f for f in H.FINGERS if f'_{f}_' in name),None)
        if f is None:continue
        normals[f]=normals.get(f,np.zeros(3))+fn*np.array(c.frame[:3])*(1 if b1==ob else -1);forces[f]=forces.get(f,0.)+fn
    support=sum(support_forces.values())
    if support<minimum_support_force_n:raise ValueError(f'Release anchor lacks physical nonrobot support: {support}N')
    if not normals:raise ValueError('No actual finger-object contact at release anchor')
    fk=mujoco.MjData(m);fk.qpos[:]=d.qpos;mujoco.mj_forward(m,fk);q0=fk.qpos[fc].copy();targets={}
    for f,n in normals.items():
        length=np.linalg.norm(n)
        if length<1e-9:continue
        targets[f]=fk.site_xpos[rig.fingertip_sites[f]].copy()+opening_m*n/length
    low=np.maximum(limits[:,0]+1e-8,q0-maximum_joint_increment_rad);high=np.minimum(limits[:,1]-1e-8,q0+maximum_joint_increment_rad)
    if np.any(high<=low):raise ValueError('Physical finger state too far beyond limits for bounded opening')
    initial=np.clip(q0,low,high)
    def residual(q):
        fk.qpos[fc]=q;mujoco.mj_forward(m,fk)
        return np.r_[np.concatenate([(fk.site_xpos[rig.fingertip_sites[f]]-p)/opening_m for f,p in targets.items()]),.1*(q-initial)/.1]
    fit=least_squares(residual,initial,bounds=(low,high),max_nfev=80)
    t=np.arange(len(Q))/fps;mask=t>start_s;u=smooth((t[mask]-start_s)/(open_complete_s-start_s))
    Q[np.ix_(mask,fc)]=(1-u[:,None])*Q[k,fc]+u[:,None]*fit.x;Q[np.ix_(mask,wc)]=Q[k,wc]
    direction=d.qpos[wc[:3]]-d.xpos[ob];direction[2]=0
    if np.linalg.norm(direction)<1e-8:raise ValueError('Need explicit withdrawal direction for vertically centered hand')
    direction/=np.linalg.norm(direction);retreat=withdrawal_horizontal_m*direction+np.array([0.,0.,withdrawal_up_m])
    u=smooth((t[mask]-open_complete_s)/(clear_s-open_complete_s));Q[np.ix_(mask,wc[:3])]+=u[:,None]*retreat
    assert np.array_equal(Q[:k+1],ref['qpos'][:k+1]);other=np.setdiff1d(np.arange(m.nq),np.r_[wc,fc]);assert np.array_equal(Q[:,other],ref['qpos'][:,other])
    metadata=dict(side=side,object=object_name,start_s=start_s,open_complete_s=open_complete_s,clear_s=clear_s,opening_m=opening_m,
        withdrawal_m=retreat.tolist(),support_forces_at_start_n=support_forces,finger_normal_forces_at_start_n=forces,
        finger_opening_targets_rad=dict(zip(rig.finger_qpos,fit.x.tolist())),finger_joint_increment_limit_rad=maximum_joint_increment_rad,
        ik_scaled_residual=float(np.linalg.norm(fit.fun)),protected_prefix_state_replay_error=prefix_error,
        source_object_targets_unchanged=True,source_clock_unchanged=True,requires_full_physical_evaluation=True)
    return Q,metadata
