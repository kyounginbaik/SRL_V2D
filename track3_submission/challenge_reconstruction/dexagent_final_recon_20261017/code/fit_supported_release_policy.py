"""Fit finite release parameters from a supported physical anchor, then test online."""
from pathlib import Path
import argparse, json, sys
import mujoco, numpy as np
from supported_release import plan_supported_release, TR, H
from release_feedback_policy import ReleasePolicy
from component_policy import ComponentPolicy
from learned_feedback_policy import sha
import evaluate_policy_component as evaluator
from audit_full_joint_limits import main as audit_joints

B = Path(__file__).resolve().parents[1]


def run(start, opening, output, place, active_only, placement_extra):
    source = B/'refinement_20261018/command_precision/episode_000003/decimals_6/node01'
    old = json.loads((source/'evaluation/result.json').read_text())
    scene = Path(old['execution']['scene']); trajectory = Path(old['execution']['trajectory'])
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/'training_report.json').exists(): raise FileExistsError(output)
    m = mujoco.MjModel.from_xml_path(str(scene)); base = ComponentPolicy(source/'policy.npz')
    rig = H.HandRig.build(m, 'robot_right_', 'right')
    wc = np.array([rig.wrist_qpos[n] for n in ['x','y','z','rx','ry','rz']])
    fc = np.array(list(rig.finger_qpos.values())); fa = np.array(list(rig.finger_act.values()))
    wa = np.array([np.flatnonzero(base.qadr == q)[0] for q in wc])
    with np.load(trajectory, allow_pickle=False) as z:
        frames = z['qpos'].copy(); times = z['timestamps_seconds']; controls = z['ctrl']; qvel = z['qvel']
        recorded_substeps = z['substep_qpos'].copy(); old_controls = z['substep_ctrl'].copy()
    # Planning-only reference of achieved controls; this file is never an actor input.
    frames[:, base.qadr] = controls
    ref = TR.save_reference(output/'planning_commands.npz', m, frames,
        ['right: supported release fitting']*len(frames), float(base.data['fps']), np.zeros(len(frames), int),
        [{'stage1_indices':[0], 'motion':'release', 'side':'right'}],
        source='Recorded control observations used only for fitting finite release parameters', arms_coupled=False)
    Q, metadata = plan_supported_release(scene, trajectory, ref, side='right', object_name='wooden_spoon',
        start_s=start, open_complete_s=start+.25, clear_s=start+.95, opening_m=opening,
        withdrawal_horizontal_m=.09, withdrawal_up_m=.07,
        minimum_support_force_n=0. if place else .1)
    k = round(start*float(base.data['fps']))
    with np.load(source/'policy.npz', allow_pickle=False) as z: weights = {key:z[key] for key in z.files}
    # The base actor at the anchor with the achieved state is the nominal point
    # for its continuously active proprioceptive feedback during opening.
    anchor = base.base if hasattr(base, 'base') else base.actor
    nominal = anchor(start, recorded_substeps[round(start/.002)-1], qvel[k])
    weights.update(release_base_module=np.array('pinky_margin_policy_v2'), release_base_class=np.array('PinkyMarginPolicy'),
        release_start_s=np.array(start), release_open_duration_s=np.array(.25), release_retreat_duration_s=np.array(.7),
        release_actuator_ids=np.r_[wa,fa], release_finger_actuator_ids=fa, release_translation_actuator_ids=wa[:3],
        release_open_delta_rad=Q[-1,fc]-nominal[fa], release_retreat_m=np.asarray(metadata['withdrawal_m']))
    if active_only:
        contact_digits=set(metadata['finger_normal_forces_at_start_n'])
        selected=np.array([name.split('_')[0] in contact_digits for name in rig.finger_qpos])
        weights.update(release_actuator_ids=np.r_[wa,fa[selected]],release_finger_actuator_ids=fa[selected],
            release_open_delta_rad=(Q[-1,fc]-nominal[fa])[selected])
        metadata['unloaded_digits_keep_original_learned_phase']=True
    if place:
        fk=mujoco.MjData(m);fk.qpos[:]=recorded_substeps[round(start/.002)-1];mujoco.mj_forward(m,fk)
        ob=m.body('wooden_spoon').id;verts=[]
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=ob or not m.geom_contype[g] or m.geom_dataid[g]<0:continue
            mid=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
            verts.extend(v@fk.geom_xmat[g].reshape(3,3).T+fk.geom_xpos[g])
        v=np.asarray(verts);lowest=v[v[:,2].argmin()];table=m.geom('g_table').id;center=fk.geom_xpos[table];top=center[2]+m.geom_size[table,1]
        inward=center[:2]-lowest[:2];inward/=np.linalg.norm(inward)
        delta=np.r_[.02*inward,-(lowest[2]-top+.003+placement_extra)]
        weights.update(release_place_duration_s=np.array(.3),release_place_delta_m=delta,release_retreat_duration_s=np.array(.65))
        metadata.update(placement_delta_m=delta.tolist(),placement_duration_s=.3,lowest_collision_vertex_m=lowest.tolist(),
            anchor_currently_unsupported=True,release_support_must_be_verified_after_physical_placement=True)
    checkpoint = output/'policy.npz'; np.savez_compressed(checkpoint, **weights)
    report = dict(method='Learned phase/proprioception actor with finite, contact-normal fitted supported-release parameters',
        source_checkpoint=str(source/'policy.npz'), source_checkpoint_sha256=sha(source/'policy.npz'),
        original_learned_weights_unchanged=True, release_fit=metadata, scene_unchanged=True,
        source_object_targets_unchanged=True, initial_state_unchanged=True,
        runtime_inputs=['episode phase','actual robot joint positions','actual robot joint velocities'],
        runtime_forbidden_inputs=['object target/state','recorded trajectory','future state'],
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
    (output/'training_report.json').write_text(json.dumps(report,indent=2)+'\n')
    evaluator.LearnedPolicy = ReleasePolicy
    result = evaluator.evaluate(3,scene,checkpoint,output/'evaluation',old['input_dir'])
    with np.load(result['execution']['trajectory'],allow_pickle=False) as z:
        end=round(start/.002)
        err=float(np.max(abs(z['substep_qpos'][:end]-recorded_substeps[:end])))
        ctrlerr=float(np.max(abs(z['substep_ctrl'][:end]-old_controls[:end])))
    if err != 0. or ctrlerr != 0.: raise AssertionError((err,ctrlerr))
    audit_joints(output/'evaluation')
    from audit_object_support import audit
    audit(output/'evaluation')
    (output/'prefix_validation.json').write_text(json.dumps(dict(until_s=start,max_qpos_error=err,max_ctrl_error=ctrlerr),indent=2)+'\n')
    print(json.dumps(result['metrics']['declared_frame_reference_proxy'],indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=float,default=14.4);p.add_argument('--opening',type=float,default=.008);p.add_argument('--output',type=Path,required=True);p.add_argument('--place',action='store_true');p.add_argument('--active-only',action='store_true');p.add_argument('--placement-extra',type=float,default=0.)
    run(**vars(p.parse_args()))
