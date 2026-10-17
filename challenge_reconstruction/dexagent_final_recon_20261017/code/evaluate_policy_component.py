# Separate evaluator for explicit versioned reconstruction adapters; archived episode8 evaluator is unchanged.
"""Fresh online evaluation of learned weights in the frozen MuJoCo scene.

The actor receives only phase and physical hand proprioception. No expert
trajectory, target object poses, reference commands, or future state enter it.
Actual controls/states are independently replayed before exporting a result.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from component_policy import ComponentPolicy as LearnedPolicy
from learned_feedback_policy import sha
from floating_runtime import _assert_contract, _write

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import episode as E
from evaluate_mp_sr import evaluate_files


def evaluate(episode,scene,checkpoint,output,episode_dir):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out/'result.json')
    execution_dir=out/'execution';execution_dir.mkdir(exist_ok=True)
    # Copy the exact XML bytes; its mesh paths are already absolute and frozen.
    frozen=execution_dir/'scene_physics.xml';shutil.copyfile(scene,frozen)
    model=mujoco.MjModel.from_xml_path(str(frozen));_assert_contract(model)
    policy=LearnedPolicy(checkpoint)
    if str(policy.data['scene_sha256'])!=sha(frozen):raise ValueError('Policy/physics mismatch')
    if policy.data['actuator_names'].tolist()!=[model.actuator(i).name for i in range(model.nu)]:
        raise ValueError('Actor action ordering mismatch')
    d=mujoco.MjData(model)
    if model.nkey:mujoco.mj_resetDataKeyframe(model,d,0)
    mujoco.mj_forward(model,d)
    state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
    initial_state=np.empty(mujoco.mj_stateSize(model,state_spec))
    mujoco.mj_getState(model,d,initial_state,state_spec)
    initial_qpos=d.qpos.copy();initial_qvel=d.qvel.copy();initial_act=d.act.copy()
    fps=float(policy.data['fps']);T=int(policy.data['frames']);dt=float(model.opt.timestep)
    if abs((T-1)/fps-policy.duration)>1e-8:raise ValueError('Checkpoint duration mismatch')
    total_steps=round(policy.duration/dt)
    qadr=policy.qadr; parts=E.robot_parts(model)
    robot={b for b in range(model.nbody) if (model.body(b).name or '').startswith('robot_')}
    side_body={b:si for si,s in enumerate(('left','right')) for b in parts[s]['rig'].body_ids}
    slide=model.jnt_type[model.actuator_trnid[:,0]]==mujoco.mjtJoint.mjJNT_SLIDE
    rotation=np.array([any((model.joint(int(j)).name or '').endswith(k) for k in ('rot_x','rot_y','rot_z')) for j in model.actuator_trnid[:,0]])
    Q=np.zeros((T,model.nq));V=np.zeros((T,model.nv));C=np.zeros((T,model.nu));times=np.zeros(T)
    commanded=np.zeros_like(Q);pen=np.zeros(T);op=np.zeros(T);forces=np.zeros((T,2));wrench=np.zeros((T,2))
    tracking=np.zeros((T,2));contacts=[[] for _ in range(T)]
    sq=np.empty((total_steps,model.nq));sv=np.empty((total_steps,model.nv));sc=np.empty((total_steps,model.nu));st=np.empty(total_steps)
    measured=mujoco.MjData(model);target=mujoco.MjData(model);force6=np.zeros(6)

    def diagnose(t):
        touched=set();net=np.zeros((2,3))
        for ci in range(d.ncon):
            c=d.contact[ci];b1,b2=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
            r1,r2=b1 in robot,b2 in robot;depth=max(0.,-float(c.dist))
            if r1!=r2:
                pen[t]=max(pen[t],depth)
                hand,obj=(b1,b2) if r1 else (b2,b1);si=side_body.get(hand)
                if si is not None:
                    touched.add((('left','right')[si],model.body(obj).name))
                    mujoco.mj_contactForce(model,d,ci,force6)
                    net[si]+=c.frame.reshape(3,3).T@force6[:3]*(1 if r1 else -1)
            elif not r1 and not r2:op[t]=max(op[t],depth)
        forces[t]=np.maximum(forces[t],np.linalg.norm(net,axis=1))
        wrench[t]=np.maximum(wrench[t],[np.max(np.abs(d.actuator_force[slide]),initial=0.),np.max(np.abs(d.actuator_force[rotation]),initial=0.)])
        return sorted(touched)

    def record(t):
        Q[t]=d.qpos;V[t]=d.qvel;C[t]=d.ctrl;times[t]=d.time
        commanded[t]=d.qpos;commanded[t,qadr]=d.ctrl
        measured.qpos[:]=d.qpos;mujoco.mj_kinematics(model,measured)
        target.qpos[:]=commanded[t];mujoco.mj_kinematics(model,target)
        for part in parts.values():
            b=part['ee_body'];p=np.linalg.norm(target.xpos[b]-measured.xpos[b])
            R=target.xmat[b].reshape(3,3).T@measured.xmat[b].reshape(3,3)
            tracking[t]=np.maximum(tracking[t],[p,np.degrees(Rotation.from_matrix(R).magnitude())])
        contacts[t]=diagnose(t)

    record(0);unstable=None;k=0;completed=1
    for t in range(1,T):
        n=round(t/(fps*dt))-round((t-1)/(fps*dt))
        for _ in range(n):
            # Causal actor inference BEFORE the physical step. No reference
            # arrays exist in this process's controller inputs.
            d.ctrl[:]=policy(d.time+dt,d.qpos,d.qvel)
            mujoco.mj_step(model,d)
            sq[k]=d.qpos;sv[k]=d.qvel;sc[k]=d.ctrl;st[k]=d.time;k+=1
            diagnose(t)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
                unstable={'frame':t,'reason':'nonfinite state or BADQACC reset'};break
        record(t);completed=t+1
        if unstable:break
    final_state=np.empty_like(initial_state);mujoco.mj_getState(model,d,final_state,state_spec)
    sl=slice(0,completed)
    diag=dict(qpos=Q[sl],qvel=V[sl],ctrl=C[sl],time_s=times[sl],initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,initial_act=initial_act,initial_state=initial_state,final_state=final_state,
        state_spec=int(state_spec),hand_pen_m=pen[sl],object_pen_m=op[sl],track_pos_m=tracking[sl,0],
        track_rot_deg=tracking[sl,1],hand_object_force_n=forces[sl],wrist_force_n=wrench[sl,0],
        wrist_torque_nm=wrench[sl,1],contacts=contacts[:completed],unstable=unstable,
        substep_qpos=sq[:k],substep_qvel=sv[:k],substep_ctrl=sc[:k],substep_time_s=st[:k])
    # Verify the recorded action really is the checkpoint output at its causal
    # pre-state. This is separate from the independent physical control replay.
    before_q=np.vstack([initial_qpos[None],sq[:k-1]])
    before_v=np.vstack([initial_qvel[None],sv[:k-1]])
    action_error=0.
    for i in range(k):
        predicted=policy(st[i],before_q[i],before_v[i])
        action_error=max(action_error,float(np.max(np.abs(predicted-sc[i]))))
    if action_error>1e-10:raise RuntimeError('Recorded controls do not match causal checkpoint inference')
    execution=_write(model,frozen,execution_dir,diag,commanded,[f'learned policy frame {i}' for i in range(T)],fps,
        dict(scope='Genuinely trained phase/proprioception imitation feedback policy; custom MuJoCo evaluator',
             learned_policy=True,policy_checkpoint=str(Path(checkpoint).resolve()),policy_checkpoint_sha256=sha(checkpoint),
             policy_inference_calls=k,causal_policy_action_reproduction_max_error=action_error,no_retiming=True,
             requested_duration_s=(T-1)/fps))
    inp=Path(episode_dir).resolve();manifest=json.loads((inp/'input_manifest.json').read_text())
    if manifest['episode'] != f'episode_{episode:06d}':raise ValueError('Versioned adapter episode mismatch')
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    result=dict(episode=f'episode_{episode:06d}',input_dir=str(inp),learned_policy=True,policy_checkpoint=str(Path(checkpoint).resolve()),
        policy_checkpoint_sha256=sha(checkpoint),execution=execution,metrics=metrics,
        policy_architecture=policy.architecture,evaluator_script_sha256=sha(__file__),policy_code_sha256=sha(Path(__file__).with_name('component_policy.py')),
        evaluation_backend='Custom frozen-scene MuJoCo500Hz; not bundled FlashSAC/official harness',
        reference_proxy_success=metrics['reference_proxy_mp_sr']==1.)
    (out/'mp_sr.json').write_text(json.dumps(metrics,indent=2)+'\n')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True)
    p.add_argument('--episode-dir',required=True);p.add_argument('--scene',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    evaluate(**vars(p.parse_args()))
