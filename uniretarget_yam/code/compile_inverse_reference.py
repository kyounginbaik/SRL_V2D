"""Compile physical native-PD feedforward into a robot-only reference tape."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf
from flash_chord.configuration import instantiate_typed
from flash_chord.envs.rl import RLEnvConfig
from flash_chord.retargeting.evaluation_task import apply_task_protocol
from flash_chord.retargeting.native_inverse_dynamics import NativeScalarPDInverse
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import CONFIG_METADATA_KEY


def checksum(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_velocity(model,q,fps):
    velocity=np.zeros((len(q),model.nv))
    for i in range(len(q)):
        a=max(0,i-1);b=min(len(q)-1,i+1)
        mujoco.mj_differentiatePos(model,velocity[i],(b-a)/fps,q[a],q[b])
        for j,kind in enumerate(model.jnt_type):
            if kind not in (0,1):continue
            qi=int(model.jnt_qposadr[j])+(3 if kind==0 else 0)
            di=int(model.jnt_dofadr[j])+(3 if kind==0 else 0)
            previous=Rotation.from_quat(q[a,qi:qi+4][[1,2,3,0]])
            current=Rotation.from_quat(q[i,qi:qi+4][[1,2,3,0]])
            velocity[i,di:di+3]=(current.inv()*previous).apply(velocity[i,di:di+3])
    return velocity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--discovery',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.plan/'manifest.json').read_text());model=mujoco.MjModel.from_binary_path(str(args.model))
    configuration=json.loads(read_checkpoint_metadata(manifest['checkpoint'])[CONFIG_METADATA_KEY])
    resolved=apply_task_protocol(configuration,manifest)
    config=instantiate_typed(OmegaConf.create(resolved['env']),RLEnvConfig)
    with np.load(args.plan/'plan.npz') as z:qbank=z['joint_q'];tbank=z['joint_target']
    with np.load(args.discovery/'physics.npz') as d:
        joint_mapping=d['native_joint_to_newton_joint'][0];qs=d['newton_joint_q_start'];ds=d['newton_joint_qd_start']
    nq=qbank.shape[2];frames=qbank.shape[1];fps=float(config.sim.fps)
    robots=[j for j in range(model.njnt) if joint_mapping[j]>=0 and qs[joint_mapping[j]]<nq]
    balls={j:(float(config.action.wrist_kp),float(config.action.wrist_tau_max)) for j in robots if model.jnt_type[j]==1}
    inverse=NativeScalarPDInverse(model,ball_drives=balls)
    names=[];geometries=[];targets=[];diagnostics=[]
    for candidate in [0,1]:
        geometry=qbank[candidate];original=tbank[candidate];native=np.repeat(model.qpos0[None],frames,axis=0)
        for j in robots:
            nj=int(joint_mapping[j]);qi=int(qs[nj]);native_qi=int(model.jnt_qposadr[j]);kind=int(model.jnt_type[j])
            width={0:7,1:4,2:1,3:1}[kind]
            values=geometry[:,qi:qi+width].copy()
            if kind==1:values=values[:,[3,0,1,2]]
            if kind==0:values=np.c_[values[:,:3],values[:,[6,3,4,5]]]
            native[:,native_qi:native_qi+width]=values
        velocity=native_velocity(model,native,fps);acceleration=np.gradient(velocity,1/fps,axis=0,edge_order=2)
        corrected=original.astype(float).copy();violations=[]
        for frame in range(frames):
            result=inverse.targets(native[frame],velocity[frame],acceleration[frame])
            for j,value in result.native_joint_position_targets.items():
                if j in robots:corrected[frame,int(ds[joint_mapping[j]])]=value
            for j,value in result.native_ball_rotvec_targets.items():
                di=int(ds[joint_mapping[j]]);corrected[frame,di:di+3]=value
            violations.extend([dict(frame=frame,**row) for row in result.violations])
        # Keep exact reset/terminal commands and ramp feedforward before the
        # first physical contact. The last row is not an evaluated transition.
        ramp=max(2,round(.25*fps));weight=np.ones(frames)
        weight[:ramp+1]=.5-.5*np.cos(np.linspace(0,np.pi,ramp+1));weight[-1]=0.
        published=original.astype(float).copy()
        scalar_targets=sorted(set(int(ds[joint_mapping[j]]) for j in inverse.position if j in robots))
        published[:,scalar_targets]+=weight[:,None]*(corrected[:,scalar_targets]-published[:,scalar_targets])
        for j in balls:
            di=int(ds[joint_mapping[j]]);initial=Rotation.from_rotvec(original[:,di:di+3].astype(float))
            corrected_r=Rotation.from_rotvec(corrected[:,di:di+3]);delta=(initial.inv()*corrected_r).as_rotvec()
            published[:,di:di+3]=(initial*Rotation.from_rotvec(weight[:,None]*delta)).as_rotvec()
        published[[0,-1]]=original[[0,-1]]
        for operation,command in [('unchanged',original),('inverse_dynamics_native_targets',published)]:
            geometries.append(geometry);targets.append(command);names.append(f'source{candidate}_{operation}')
        diagnostics.append(dict(source_candidate=candidate,actuator_violations=violations,
            peak_native_target_change=float(np.max(np.abs(published-original))),
            scalar_joints=len(scalar_targets),ball_joints=len(balls)))
    qs_out=np.asarray(geometries);ts_out=np.asarray(targets,dtype=np.float32)
    np.testing.assert_array_equal(qs_out[:,[0,-1]],qbank[0,[0,-1]][None].repeat(len(qs_out),axis=0))
    np.savez_compressed(args.output/'plan.npz',joint_q=qs_out,joint_target=ts_out)
    manifest.update(candidates=[dict(name=name,controller='zero residual native targets') for name in names],
        inverse_dynamics_diagnostics=diagnostics,source_plan=str(args.plan.resolve()),
        source_plan_sha256=checksum(args.plan/'plan.npz'),native_model_sha256=checksum(args.model),
        no_object_force_or_state_assistance=True,geometry_unchanged=True,
        compiled_plan_sha256=checksum(args.output/'plan.npz'))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(candidates=names,diagnostics=diagnostics)),flush=True)

if __name__=='__main__':main()
