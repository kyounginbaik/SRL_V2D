"""Bounded motor-target feedforward, without external force application.

For a position servo F=kp*(u-q)-kv*qdot, choose
u=q_reference+(gravity/Coriolis_bias+kv*qdot_reference)/kp.
Only wrist target coordinates change; the actuator still enforces its original
force limits. Bias is MuJoCo qfrc_bias (no contact/constraint inverse dynamics).
"""
from pathlib import Path
import argparse,json,sys
import numpy as np,mujoco
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from floating_runtime import execute_reference
from calibrate_controller import tracking_audit
from evaluate_mp_sr import evaluate_files


def make_reference(model,source,destination,mode='gravity_velocity'):
    ref=dict(np.load(source));q=ref['qpos'].copy();fps=float(ref['fps']);T=len(q)
    v=np.zeros((T,model.nv));one=np.zeros((T-1,model.nv))
    for t in range(T-1):mujoco.mj_differentiatePos(model,one[t],1/fps,q[t],q[t+1])
    v[0]=one[0];v[-1]=one[-1];v[1:-1]=.5*(one[:-1]+one[1:])
    aids=[a for a in range(model.nu) if model.joint(int(model.actuator_trnid[a,0])).name.endswith(tuple('_'+s for s in ('pos_x','pos_y','pos_z','rot_x','rot_y','rot_z')))]
    jids=model.actuator_trnid[aids,0];qadr=model.jnt_qposadr[jids];dadr=model.jnt_dofadr[jids]
    kp=model.actuator_gainprm[aids,0];kv=-model.actuator_biasprm[aids,2]
    d=mujoco.MjData(model);offset=np.zeros((T,len(aids)));clipped_force_samples=0
    for t in range(1,T):
        ff=np.zeros(len(aids))
        if 'velocity' in mode:ff+=kv*v[t,dadr]
        if 'gravity' in mode:
            d.qpos[:]=q[t];d.qvel[:]=v[t] if 'velocity' in mode else 0.
            mujoco.mj_forward(model,d)
            ff+=d.qfrc_bias[dadr]
        clipped=np.clip(ff,model.actuator_forcerange[aids,0],model.actuator_forcerange[aids,1])
        clipped_force_samples+=int(np.sum(clipped!=ff));offset[t]=clipped/kp
    result=q.copy();result[:,qadr]=np.clip(q[:,qadr]+offset,model.actuator_ctrlrange[aids,0],model.actuator_ctrlrange[aids,1])
    others=np.setdiff1d(np.arange(model.nq),qadr)
    assert np.array_equal(result[:,others],q[:,others]);assert np.array_equal(result[0],q[0])
    ref['qpos']=result;np.savez_compressed(destination,**ref)
    return dict(mode=mode,force_caps_unchanged=True,object_reference_unchanged=True,finger_reference_unchanged=True,
        initial_state_unchanged=True,actuation='position target offsets only; no external forces',
        offset_force_clipped_samples=clipped_force_samples,
        max_absolute_offset_by_joint={model.joint(j).name:float(np.max(np.abs(offset[:,i]))) for i,j in enumerate(jids)})


def run(task):
    candidate=task//3;mode=['gravity','velocity','gravity_velocity'][task%3]
    source=B/f'refinement_20261017/lid_pressure/episode_000001/candidate_{candidate:02d}_press_1/reference.npz'
    scene=source.parent/'execution/scene_physics.xml'
    out=B/f'refinement_20261017/wrist_feedforward/episode_000001/candidate_{candidate:02d}_{mode}';out.mkdir(parents=True,exist_ok=True)
    model=mujoco.MjModel.from_xml_path(str(scene));reference=out/'reference.npz'
    adjustment=make_reference(model,source,reference,mode)
    execution=execute_reference(scene,reference,out/'execution')
    metrics=evaluate_files(B.parent/'final_recon/episode_000001',execution['trajectory'],B/'episodes/episode_000001/input_manifest.json',execution['scene'])
    report=dict(candidate=candidate,feedforward=adjustment,execution=execution,metrics=metrics,
        desired_tracking=tracking_audit(execution['scene'],execution['trajectory'],source))
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',type=int,required=True);a=p.parse_args();run(a.task)
