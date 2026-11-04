"""Delay finger opening while retaining the learned wrist setdown motion."""
from pathlib import Path
import argparse,json
import numpy as np
from evaluate_policy_component import evaluate
from learned_feedback_policy import sha
from audit_joint_limits import audit as joint_audit
from audit_object_support import audit as support_audit
B=Path(__file__).resolve().parents[1]

def run(case):
    source=B/'refinement_20261018/expanded_batch/bimanual_release/episode_000006/case_0';old=json.loads((source/'evaluation/result.json').read_text());out=B/f'refinement_20261018/expanded_batch/finger_release6/case_{case}';out.mkdir(parents=True,exist_ok=True)
    if (out/'training_report.json').exists():raise FileExistsError(out)
    with np.load(source/'policy.npz',allow_pickle=False) as z:w={k:z[k] for k in z.files}
    p=str(w['multi_release_prefixes'][-1]);delay=[.15,.35,.50,.15,.35,.50][case];duration=.15 if case<3 else .3
    w[p+'actuator_ids']=w[p+'finger_actuator_ids'].copy();w[p+'translation_actuator_ids']=np.array([],int);w[p+'rotation_actuator_ids']=np.array([],int)
    w[p+'place_duration_s']=np.array(delay);w[p+'place_delta_m']=np.array([],float);w[p+'retreat_m']=np.array([],float);w[p+'open_duration_s']=np.array(duration)
    w.pop(p+'place_rotation_vector',None);policy=out/'policy.npz';np.savez_compressed(policy,**w)
    result=evaluate(6,old['execution']['scene'],policy,out/'evaluation',old['input_dir']);quality=joint_audit(result['execution']['scene'],result['execution']['trajectory']);ad=out/'evaluation/audit';ad.mkdir(exist_ok=True);(ad/'actual_joint_limits.json').write_text(json.dumps(quality,indent=2)+'\n');support_audit(out/'evaluation')
    report=dict(method='Finite finger-only release delay; learned wrist phase remains active',learned_policy=True,
        source_checkpoint=str(source/'policy.npz'),source_checkpoint_sha256=sha(source/'policy.npz'),checkpoint=str(policy),checkpoint_sha256=sha(policy),trainer_sha256=sha(__file__),
        release_delay_s=delay,opening_duration_s=duration,learned_wrist_commands_unmodified=True,original_learned_weights_unchanged=True,
        scene_clock_object_targets_and_physics_limits_unchanged=True,policy_contains_command_table=False,
        reference_proxy_success=result['reference_proxy_success'],actual_joint_max_degrees=quality['hinge_max_violation_degrees'])
    (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,choices=range(6),required=True);run(**vars(p.parse_args()))
