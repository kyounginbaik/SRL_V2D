"""Open the achieved blue-cup grasp before withdrawing along a clear direction.

The successful pickup/carry prefix, objects, clock and physical model are fixed.
Only robot release commands are changed; no object pose or force is injected.
"""
from pathlib import Path
import json,sys
import mujoco,numpy as np
from scipy.optimize import least_squares

B=Path(__file__).resolve().parents[1];R=B/'refinement_20261018/episode_000010'
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H,tracking as TR
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from audit_full_joint_limits import main as audit_limits


def smooth(x):
    x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)


def main():
    source=R/'calibrated_blue_pickup';out=R/'calibrated_blue_release';out.mkdir(exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    scene=source/'execution/scene_physics.xml';m=mujoco.MjModel.from_xml_path(str(scene));ref=TR.load_reference(source/'reference.npz',m);Q=ref['qpos'].copy();fps=ref['fps']
    z=np.load(source/'execution/trajectory.npz');d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['initial_state'],int(z['state_spec']))
    start=5.5;opened_at=5.85;clear_at=6.8;step=round(start/m.opt.timestep)
    for ctrl in z['substep_ctrl'][:step]:d.ctrl[:]=ctrl;mujoco.mj_step(m,d)
    rig=H.HandRig.build(m,'robot_left_','left');wc=np.array([rig.wrist_qpos[n] for n in ['x','y','z','rx','ry','rz']]);fc=np.array(list(rig.finger_qpos.values()));aids=np.array(list(rig.finger_act.values()));limits=m.jnt_range[m.actuator_trnid[aids,0]];ob=m.body('blue_cup').id
    normals={};forces={};force=np.zeros(6);support=0.
    for i,c in enumerate(d.contact):
        b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);other=b2 if b1==ob else b1 if b2==ob else -1
        if other<0:continue
        mujoco.mj_contactForce(m,d,i,force);fn=max(0.,float(force[0]))
        if not m.body(other).name.startswith('robot_'):support+=fn
        if other not in rig.body_ids or fn<.1:continue
        f=next((f for f in H.FINGERS if f'_{f}_' in m.body(other).name),None)
        if f is None:continue
        normals[f]=normals.get(f,np.zeros(3))+fn*np.array(c.frame[:3])*(1 if b1==ob else -1);forces[f]=forces.get(f,0.)+fn
    if support<.1:raise ValueError('Release anchor is not supported by table/other objects')
    fk=mujoco.MjData(m);fk.qpos[:]=d.qpos;mujoco.mj_forward(m,fk);q0=fk.qpos[fc].copy();targets={}
    for f,n in normals.items():
        n/=np.linalg.norm(n);targets[f]=fk.site_xpos[rig.fingertip_sites[f]].copy()+.008*n
    low=np.maximum(limits[:,0]+1e-8,q0-.35);high=np.minimum(limits[:,1]-1e-8,q0+.35);initial=np.clip(q0,low,high)
    def residual(q):
        fk.qpos[fc]=q;mujoco.mj_forward(m,fk)
        return np.r_[np.concatenate([(fk.site_xpos[rig.fingertip_sites[f]]-p)/.008 for f,p in targets.items()]),.1*(q-initial)/.1]
    fit=least_squares(residual,initial,bounds=(low,high),max_nfev=80)
    k=round(start*fps);t=np.arange(len(Q))/fps;mask=t>start;u=smooth((t[mask]-start)/(opened_at-start))
    Q[np.ix_(mask,fc)]=(1-u[:,None])*Q[k,fc]+u[:,None]*fit.x
    Q[np.ix_(mask,wc)]=Q[k,wc]
    direction=d.qpos[wc[:3]]-d.xpos[ob];direction[2]=0;direction/=np.linalg.norm(direction);retreat=.12*direction+np.array([0.,0.,.08])
    u=smooth((t[mask]-opened_at)/(clear_at-opened_at));Q[np.ix_(mask,wc[:3])]+=u[:,None]*retreat
    assert np.array_equal(Q[:k+1],ref['qpos'][:k+1])
    other=np.setdiff1d(np.arange(m.nq),np.r_[wc,fc]);assert np.array_equal(Q[:,other],ref['qpos'][:,other])
    reference=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source='Actual contact-normal finger opening, then collision-clear robot withdrawal after source-timed setdown; successful pickup and all object targets unchanged.',floating_runtime=True)
    execution=execute_reference(scene,reference,out/'execution');inp=R/'calibrated_inputs/episodes/episode_000010';manifest=json.loads((inp/'input_manifest.json').read_text());metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    new=np.load(execution['trajectory']);prefixerr=float(abs(new['substep_qpos'][:step]-z['substep_qpos'][:step]).max());assert prefixerr==0.
    report=dict(episode='episode_000010',reference=str(reference),execution=execution,metrics=metrics,input_manifest=str(inp/'input_manifest.json'),source_clock_preserved=True,object_targets_unchanged=True,
        protected_prefix_s=start,protected_prefix_max_qpos_error=prefixerr,release=dict(start_s=start,open_complete_s=opened_at,withdraw_complete_s=clear_at,nonrobot_support_force_at_start_n=support,contact_forces_at_start_n=forces,withdrawal_m=retreat.tolist(),finger_opening_target_m=.008,ik_final_residual=float(np.linalg.norm(fit.fun))),learned_policy=False)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');audit_limits(out)
    print(json.dumps(dict(mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['declared_frame_reference_proxy']['objects'])),flush=True)


if __name__=='__main__':main()
