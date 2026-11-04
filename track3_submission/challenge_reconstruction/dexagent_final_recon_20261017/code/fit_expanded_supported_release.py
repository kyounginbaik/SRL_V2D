"""Fit robot-only release parameters while preserving a learned carry prefix."""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from supported_release import plan_supported_release,TR,H
from component_policy import ComponentPolicy
from learned_feedback_policy import sha
from evaluate_policy_component import evaluate
from audit_joint_limits import audit as joint_audit
from audit_object_support import audit as support_audit

B=Path(__file__).resolve().parents[1]

def run(episode,start,opening):
    source=B/f'refinement_20261018/expanded_batch/feasible_hand_margin_policy/episode_{episode:06d}'
    old=json.loads((source/'shared_evaluation/result.json').read_text());scene=Path(old['execution']['scene']);trajectory=Path(old['execution']['trajectory'])
    out=B/f'refinement_20261018/expanded_batch/learned_supported_release_v2/episode_{episode:06d}/start_{start:.2f}_open_{opening:.3f}'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'training_report.json').exists():raise FileExistsError(out)
    object_name='blue_cup' if episode==33 else 'white_pot';side='left'
    m=mujoco.MjModel.from_xml_path(str(scene));base=ComponentPolicy(source/'policy.npz');rig=H.HandRig.build(m,'robot_left_',side)
    wc=np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]);fc=np.array(list(rig.finger_qpos.values()));fa=np.array(list(rig.finger_act.values()));wa=np.array([rig.wrist_act[k] for k in ['x','y','z','rx','ry','rz']])
    with np.load(trajectory,allow_pickle=False) as z:
        frames=z['qpos'].copy();controls=z['ctrl'].copy();sq=z['substep_qpos'].copy();sv=z['substep_qvel'].copy();sc=z['substep_ctrl'].copy();st=z['substep_time_s'].copy()
    frames[:,base.qadr]=controls
    reference=TR.save_reference(out/'planning_commands.npz',m,frames,[f'{side}: supported release fit']*len(frames),float(base.data['fps']),np.zeros(len(frames),int),[{'stage1_indices':[0],'motion':'release','side':side}],source='Recorded physical commands for fitting finite release parameters only',arms_coupled=False)
    Q,meta=plan_supported_release(scene,trajectory,reference,side=side,object_name=object_name,start_s=start,open_complete_s=start+.3,clear_s=start+1.,opening_m=opening)
    end=round(start/m.opt.timestep)
    # Gate after the actual last preserved integration timestamp. Decimal source
    # frame times can round just below it after thousands of .002 s additions.
    gate=float(np.nextafter(st[end-1],np.inf));nominal=base(gate,sq[end-1],sv[end-1])
    with np.load(source/'policy.npz',allow_pickle=False) as z:weights={k:z[k] for k in z.files}
    weights.update(release_base_module=np.array('pinky_margin_policy_v2'),release_base_class=np.array('PinkyMarginPolicy'),
        release_start_s=np.array(gate),release_open_duration_s=np.array(.3),release_retreat_duration_s=np.array(.7),
        release_actuator_ids=np.r_[wa,fa],release_finger_actuator_ids=fa,release_translation_actuator_ids=wa[:3],
        release_open_delta_rad=Q[-1,fc]-nominal[fa],release_retreat_m=np.asarray(meta['withdrawal_m']))
    policy=out/'policy.npz';np.savez_compressed(policy,**weights)
    result=evaluate(episode,scene,policy,out/'evaluation',old['input_dir'])
    with np.load(result['execution']['trajectory']) as z:
        errors=dict(qpos=float(np.max(abs(z['substep_qpos'][:end]-sq[:end]))),ctrl=float(np.max(abs(z['substep_ctrl'][:end]-sc[:end]))))
    if max(errors.values())!=0.:raise AssertionError(errors)
    joint=joint_audit(result['execution']['scene'],result['execution']['trajectory']);auditdir=out/'evaluation/audit';auditdir.mkdir(exist_ok=True)
    (auditdir/'actual_joint_limits.json').write_text(json.dumps(joint,indent=2)+'\n');support_audit(out/'evaluation')
    report=dict(learned_policy=True,method='Finite supported-release parameters fitted to a physical learned-policy anchor',
        source_checkpoint=str(source/'policy.npz'),source_checkpoint_sha256=sha(source/'policy.npz'),checkpoint=str(policy),checkpoint_sha256=sha(policy),
        trainer_sha256=sha(__file__),release_fit=meta,unchanged_carry_prefix_until_s=start,exact_phase_gate_s=gate,prefix_max_errors=errors,
        source_objects_clock_scene_and_physics_limits_unchanged=True,original_learned_weights_unchanged=True,policy_contains_command_table=False,
        runtime_inputs=['phase','actual robot joint positions','actual robot joint velocities'],
        reference_proxy_success=result['reference_proxy_success'],actual_joint_max_degrees=joint['hinge_max_violation_degrees'])
    (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,choices=[6,33],required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--opening',type=float,default=.008);run(**vars(p.parse_args()))
