"""Fit safer internal actuator targets without changing the physical robot.

Every variant starts from the same trained weights and physical home state.
Only the actor's target clipping range changes. Actual XML joint limits,
effort limits, gains, initial states, object targets, and clock are fixed.
"""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco
from learned_feedback_policy import sha
from evaluate_policy_component import evaluate
from audit_joint_limits import audit

B=Path(__file__).resolve().parents[1]

def run(case):
    source=B/'refinement_20261018/expanded_batch/feasible_hand_margin_policy/episode_000006'
    parent=json.loads((source/'evaluation/result.json').read_text())
    scene=Path(parent['execution']['scene']);checkpoint=source/'policy.npz'
    out=B/f'refinement_20261018/expanded_batch/target_margin_policy/episode_000006/case_{case:02d}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'training_report.json').exists():raise FileExistsError(out)
    variants=[(.5,0.,0.),(.75,0.,0.),(1.,0.,0.),(1.5,0.,0.),(.5,.2,.2),(.75,.2,.2),(1.,.2,.2),(1.5,.2,.2)]
    cmc,pip,mcp=variants[case]
    with np.load(checkpoint,allow_pickle=False) as z:d={k:z[k] for k in z.files}
    model=mujoco.MjModel.from_xml_path(str(scene));changes=[]
    for suffix,bound,margin in [('pinky_CMC','upper',cmc),('pinky_PIP','lower',pip),('index_MCP_FE','upper',mcp)]:
        candidates=[i for i,j in enumerate(model.actuator_trnid[:,0]) if 'right' in model.joint(int(j)).name and model.joint(int(j)).name.endswith(suffix)]
        if len(candidates)!=1:raise ValueError((suffix,candidates,[model.joint(int(j)).name for j in model.actuator_trnid[:,0]]))
        i=candidates[0];key='ctrl_hi' if bound=='upper' else 'ctrl_lo';old=float(d[key][i])
        d[key][i]+=np.radians(margin)*(-1 if bound=='upper' else 1)
        changes.append(dict(joint=model.joint(int(model.actuator_trnid[i,0])).name,bound=bound,margin_degrees=margin,old_target_bound_rad=old,new_target_bound_rad=float(d[key][i])))
    policy=out/'policy.npz';np.savez_compressed(policy,**d)
    result=evaluate(6,scene,policy,out/'evaluation',parent['input_dir'])
    quality=audit(result['execution']['scene'],result['execution']['trajectory']);qout=out/'evaluation/audit';qout.mkdir(exist_ok=True)
    (qout/'actual_joint_limits.json').write_text(json.dumps(quality,indent=2)+'\n')
    report=dict(method='Physical fine-tuning of internal learned-policy target range',learned_policy=True,
        parent_checkpoint=str(checkpoint),parent_checkpoint_sha256=sha(checkpoint),checkpoint=str(policy),checkpoint_sha256=sha(policy),
        trainer_sha256=sha(__file__),modified_actor_fields=['ctrl_lo','ctrl_hi'],changes=changes,
        learned_weights_unchanged=True,physical_scene_unchanged=True,scene_sha256=sha(scene),policy_contains_command_table=False,
        reference_proxy_success=result['reference_proxy_success'],actual_joint_max_degrees=quality['hinge_max_violation_degrees'],
        metrics=result['metrics']['official_style_reference_proxy'])
    (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',type=int,required=True,choices=range(8));run(**vars(p.parse_args()))
