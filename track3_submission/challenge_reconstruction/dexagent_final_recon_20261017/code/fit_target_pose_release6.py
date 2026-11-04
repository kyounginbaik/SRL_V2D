"""Fit a spoon placement transform from the achieved grasp, without runtime GT."""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from multi_release_feedback_policy import MultiReleasePolicy
from supported_release import TR
from learned_feedback_policy import sha
from audit_joint_limits import audit as joint_audit
from audit_object_support import audit as support_audit
from evaluate_policy_component import evaluate
B=Path(__file__).resolve().parents[1]

def run(case):
    source=B/'refinement_20261018/expanded_batch/bimanual_release/episode_000006/case_0';old=json.loads((source/'evaluation/result.json').read_text())
    out=B/f'refinement_20261018/expanded_batch/target_pose_release6/case_{case}';out.mkdir(parents=True,exist_ok=True)
    if (out/'training_report.json').exists():raise FileExistsError(out)
    policy=MultiReleasePolicy(source/'policy.npz');m=mujoco.MjModel.from_xml_path(old['execution']['scene']);w={k:v.copy() for k,v in policy.data.items()};p=str(w['multi_release_prefixes'][-1]);start=float(w[p+'start_s']);end=round(start/m.opt.timestep)
    with np.load(old['execution']['trajectory']) as z:q=z['substep_qpos'][end-1].copy();v=z['substep_qvel'][end-1].copy();sq=z['substep_qpos'].copy();sc=z['substep_ctrl'].copy()
    ref=TR.load_reference(Path(old['input_dir'])/'tracking_reference.npz',m);oc=int(m.joint('wooden_spoon_free').qposadr[0]);wa=w[p+'translation_actuator_ids'];ra=w[p+'rotation_actuator_ids'];wc=policy.qadr[wa];rc=policy.qadr[ra]
    target_s=[9.5,9.5,policy.duration,policy.duration][case];duration=[.35,.60,.35,.60][case]
    target=ref['qpos'][min(len(ref['qpos'])-1,round(target_s*ref['fps']))]
    currentR=Rotation.from_quat(q[oc+3:oc+7],scalar_first=True);targetR=Rotation.from_quat(target[oc+3:oc+7],scalar_first=True);delta=targetR*currentR.inv();wristR=Rotation.from_euler('XYZ',q[rc])
    newp=target[oc:oc+3]+delta.apply(q[wc]-q[oc:oc+3]);translation=newp-q[wc]
    rotation=(wristR.inv()*delta*wristR).as_rotvec()
    w[p+'place_delta_m']=translation;w[p+'place_rotation_vector']=rotation;w[p+'place_duration_s']=np.array(duration)
    checkpoint=out/'policy.npz';np.savez_compressed(checkpoint,**w)
    result=evaluate(6,old['execution']['scene'],checkpoint,out/'evaluation',old['input_dir'])
    with np.load(result['execution']['trajectory']) as z:errors=dict(qpos=float(np.max(abs(z['substep_qpos'][:end]-sq[:end]))),ctrl=float(np.max(abs(z['substep_ctrl'][:end]-sc[:end]))))
    if max(errors.values())>1e-10:raise AssertionError(errors)
    quality=joint_audit(result['execution']['scene'],result['execution']['trajectory']);ad=out/'evaluation/audit';ad.mkdir(exist_ok=True);(ad/'actual_joint_limits.json').write_text(json.dumps(quality,indent=2)+'\n');support_audit(out/'evaluation')
    report=dict(method='Finite placement transform fitted from achieved object-relative grasp to the unchanged demonstrated setdown pose',learned_policy=True,
        source_checkpoint=str(source/'policy.npz'),source_checkpoint_sha256=sha(source/'policy.npz'),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),trainer_sha256=sha(__file__),
        target_source_time_s=target_s,placement_duration_s=duration,translation_m=translation.tolist(),rotation_vector_rad=rotation.tolist(),
        prefix_until_s=start,prefix_max_errors=errors,original_learned_weights_unchanged=True,object_targets_clock_initial_state_and_physics_unchanged=True,
        runtime_inputs=['phase','robot positions','robot velocities'],policy_contains_object_trajectory=False,policy_contains_command_table=False,
        reference_proxy_success=result['reference_proxy_success'],actual_joint_max_degrees=quality['hinge_max_violation_degrees'])
    (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,choices=range(4),required=True);run(**vars(p.parse_args()))
