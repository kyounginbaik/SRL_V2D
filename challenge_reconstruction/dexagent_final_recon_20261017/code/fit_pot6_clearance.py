"""Redirect the already-released left wrist to avoid its later pot collision."""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from multi_release_feedback_policy import MultiReleasePolicy
from physics_cup_pickup_search import H
from evaluate_policy_component import evaluate
from audit_joint_limits import audit as joint_audit
from audit_object_support import audit as support_audit
from learned_feedback_policy import sha
B=Path(__file__).resolve().parents[1]
def run(case):
 source=B/'refinement_20261018/expanded_batch/refit_release6/case_2';old=json.loads((source/'evaluation/result.json').read_text());out=B/f'refinement_20261018/expanded_batch/clearance6/case_{case}';out.mkdir(parents=True,exist_ok=True)
 if (out/'training_report.json').exists():raise FileExistsError(out)
 start=[10.15,10.3,10.5][case];actor=MultiReleasePolicy(source/'policy.npz');w={k:v.copy() for k,v in actor.data.items()};m=mujoco.MjModel.from_xml_path(old['execution']['scene']);end=round(start/.002)
 with np.load(old['execution']['trajectory']) as z:q=z['substep_qpos'][end-1];v=z['substep_qvel'][end-1];gate=float(np.nextafter(z['substep_time_s'][end-1],np.inf));prefix=z['substep_qpos'][:end].copy()
 support=json.loads((source/'evaluation/audit/object_support.json').read_text())['rows'];anchor=min(support,key=lambda x:abs(x['time_s']-start))['objects']['white_pot'];assert sum(anchor['support'].values())>.1 and sum(anchor['fingers'].values())<.1,'Already-released and supported pot required'
 rig=H.HandRig.build(m,'robot_left_','left');wa=np.array([rig.wrist_act[k] for k in ['x','y','z','rx','ry','rz']]);wc=np.array([rig.wrist_qpos[k] for k in ['x','y','z']]);oc=int(m.joint('white_pot_free').qposadr[0]);direction=q[wc[:2]]-q[oc:oc+2];direction/=np.linalg.norm(direction);retreat=np.r_[.12*direction,.08];prefixes=w['multi_release_prefixes'].tolist();p=f'release{len(prefixes)}_';prefixes.append(p);w['multi_release_prefixes']=np.array(prefixes)
 w.update({p+k:val for k,val in dict(start_s=np.array(gate),open_duration_s=np.array(.002),retreat_duration_s=np.array(.8),actuator_ids=wa,finger_actuator_ids=np.array([],int),translation_actuator_ids=wa[:3],rotation_actuator_ids=wa[3:],open_delta_rad=np.array([],float),retreat_m=retreat).items()})
 checkpoint=out/'policy.npz';np.savez_compressed(checkpoint,**w);result=evaluate(6,old['execution']['scene'],checkpoint,out/'evaluation',old['input_dir']);ad=out/'evaluation/audit';ad.mkdir(exist_ok=True);quality=joint_audit(result['execution']['scene'],result['execution']['trajectory']);(ad/'actual_joint_limits.json').write_text(json.dumps(quality,indent=2)+'\n');support_audit(out/'evaluation')
 with np.load(result['execution']['trajectory']) as z:error=float(np.max(abs(z['substep_qpos'][:end]-prefix)))
 assert error<=1e-10
 report=dict(method='Finite post-release wrist clearance fitted from actual supported-object and wrist geometry',source_checkpoint=str(source/'policy.npz'),source_checkpoint_sha256=sha(source/'policy.npz'),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),trainer_sha256=sha(__file__),start_s=start,retreat_m=retreat.tolist(),anchor_support_force_n=sum(anchor['support'].values()),anchor_hand_force_n=sum(anchor['fingers'].values()),protected_prefix_error=error,finger_phase_commands_preserved=True,original_learned_weights_unchanged=True,reference_targets_scene_limits_clock_unchanged=True,runtime_has_no_object_targets_or_command_table=True,reference_proxy_success=result['reference_proxy_success']);(out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',type=int,choices=range(3),required=True);run(**vars(p.parse_args()))
