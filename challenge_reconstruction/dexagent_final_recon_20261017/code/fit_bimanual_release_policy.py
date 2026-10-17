"""Fit an additional supported robot release while retaining a learned prefix."""
from pathlib import Path
import argparse,json
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from supported_release import plan_supported_release,TR,H
from component_policy import ComponentPolicy
from multi_release_feedback_policy import MultiReleasePolicy
from learned_feedback_policy import sha
from audit_joint_limits import audit as joint_audit
from audit_object_support import audit as support_audit
import evaluate_policy_component as evaluator
B=Path(__file__).resolve().parents[1]

def run(source,output,side,object_name,start,opening=.008,place=False,place_duration=.35,rotate=0.,inward=.01):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    if (out/'training_report.json').exists():raise FileExistsError(out)
    old=json.loads((source/'evaluation/result.json').read_text());scene=Path(old['execution']['scene']);trajectory=Path(old['execution']['trajectory'])
    checkpoint=source/'policy.npz'
    with np.load(checkpoint,allow_pickle=False) as z:weights={k:z[k] for k in z.files}
    base=MultiReleasePolicy(checkpoint) if 'multi_release_prefixes' in weights else ComponentPolicy(checkpoint)
    m=mujoco.MjModel.from_xml_path(str(scene));rig=H.HandRig.build(m,f'robot_{side}_',side)
    wc=np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']]);fc=np.array(list(rig.finger_qpos.values()));fa=np.array(list(rig.finger_act.values()));wa=np.array([rig.wrist_act[k] for k in ['x','y','z','rx','ry','rz']])
    with np.load(trajectory,allow_pickle=False) as z:
        frames=z['qpos'].copy();controls=z['ctrl'].copy();sq=z['substep_qpos'].copy();sv=z['substep_qvel'].copy();sc=z['substep_ctrl'].copy();st=z['substep_time_s'].copy()
    frames[:,base.qadr]=controls
    reference=TR.save_reference(out/'planning_commands.npz',m,frames,[f'{side}: fitted release']*len(frames),float(base.data['fps']),np.zeros(len(frames),int),[{'stage1_indices':[0],'motion':'release','side':side}],source='Physical observations for fitting finite release parameters only',arms_coupled=False)
    Q,meta=plan_supported_release(scene,trajectory,reference,side=side,object_name=object_name,start_s=start,open_complete_s=start+.3,clear_s=start+1.,opening_m=opening,minimum_support_force_n=0. if place else .1)
    end=round(start/m.opt.timestep);gate=float(np.nextafter(st[end-1],np.inf));nominal=base(gate,sq[end-1],sv[end-1])
    prefixes=weights.get('multi_release_prefixes',np.array([],dtype=str)).tolist()
    if not prefixes and 'release_base_module' in weights:
        p='release0_';prefixes.append(p)
        for k,v in list(weights.items()):
            if k.startswith('release_') and k not in ['release_base_module','release_base_class']:weights[p+k[len('release_'):]]=v.copy()
    p=f'release{len(prefixes)}_';prefixes.append(p)
    weights.update(multi_release_base_module=np.array('pinky_margin_policy_v2'),multi_release_prefixes=np.array(prefixes))
    new=dict(start_s=np.array(gate),open_duration_s=np.array(.3),retreat_duration_s=np.array(.7),actuator_ids=np.r_[wa,fa],
        finger_actuator_ids=fa,translation_actuator_ids=wa[:3],rotation_actuator_ids=wa[3:],open_delta_rad=Q[-1,fc]-nominal[fa],retreat_m=np.asarray(meta['withdrawal_m']))
    if place:
        fk=mujoco.MjData(m);fk.qpos[:]=sq[end-1];mujoco.mj_forward(m,fk);ob=m.body(object_name).id;vertices=[]
        for g in range(m.ngeom):
            if m.geom_bodyid[g]!=ob or not m.geom_contype[g] or m.geom_dataid[g]<0:continue
            mid=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
            vertices.extend(v@fk.geom_xmat[g].reshape(3,3).T+fk.geom_xpos[g])
        vertices=np.asarray(vertices);lowest=vertices[vertices[:,2].argmin()];tg=m.geom('g_table').id;center=fk.geom_xpos[tg];top=center[2]+m.geom_size[tg,1]
        direction=center[:2]-lowest[:2];direction/=np.linalg.norm(direction)
        delta=np.r_[inward*direction,-(lowest[2]-top+.002)]
        new.update(place_duration_s=np.array(place_duration),place_delta_m=delta)
        if rotate:
            ref=TR.load_reference(Path(old['input_dir'])/'tracking_reference.npz',m);oc=int(m.joint(object_name+'_free').qposadr[0]);targetR=Rotation.from_quat(ref['qpos'][-1,oc+3:oc+7],scalar_first=True).as_matrix();objectR=fk.xmat[ob].reshape(3,3);wristR=Rotation.from_euler('XYZ',nominal[wa[3:]]).as_matrix()
            new['place_rotation_vector']=rotate*Rotation.from_matrix(wristR.T@(targetR@objectR.T)@wristR).as_rotvec()
        meta.update(placement_delta_m=delta.tolist(),placement_duration_s=place_duration,lowest_collision_vertex_m=lowest.tolist(),
            placement_before_opening_required=True,support_after_placement_requires_physical_verification=True,rotation_fraction=rotate)
    weights.update({p+k:v for k,v in new.items()});policy=out/'policy.npz';np.savez_compressed(policy,**weights)
    # A trial evaluator is explicit until this new architecture is registered
    # in the shared submission factory and replayed again for final packaging.
    evaluator.LearnedPolicy=MultiReleasePolicy
    ep=int(old['episode'].split('_')[-1]);result=evaluator.evaluate(ep,scene,policy,out/'evaluation',old['input_dir'])
    with np.load(result['execution']['trajectory']) as z:
        errors=dict(qpos=float(np.max(abs(z['substep_qpos'][:end]-sq[:end]))),ctrl=float(np.max(abs(z['substep_ctrl'][:end]-sc[:end]))))
    if max(errors.values())>1e-10:raise AssertionError(errors)
    quality=joint_audit(result['execution']['scene'],result['execution']['trajectory']);auditdir=out/'evaluation/audit';auditdir.mkdir(exist_ok=True)
    (auditdir/'actual_joint_limits.json').write_text(json.dumps(quality,indent=2)+'\n');support_audit(out/'evaluation')
    report=dict(learned_policy=True,method='Finite per-hand placement and release parameters fitted to physical learned-policy anchors',
        source_checkpoint=str(checkpoint),source_checkpoint_sha256=sha(checkpoint),checkpoint=str(policy),checkpoint_sha256=sha(policy),
        trainer_sha256=sha(__file__),actor_source_sha256=sha(Path(__file__).with_name('multi_release_feedback_policy.py')),
        release_fit=meta,protected_prefix_until_s=start,prefix_max_errors=errors,exact_phase_gate_s=gate,
        source_object_targets_clock_scene_and_physics_limits_unchanged=True,original_learned_weights_unchanged=True,
        runtime_inputs=['phase','measured robot positions','measured robot velocities'],policy_contains_command_table=False,
        final_shared_registry_evaluation_required=True,reference_proxy_success=result['reference_proxy_success'],actual_joint_max_degrees=quality['hinge_max_violation_degrees'])
    (out/'training_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--side',choices=['left','right'],required=True);p.add_argument('--object-name',required=True);p.add_argument('--start',type=float,required=True);p.add_argument('--opening',type=float,default=.008);p.add_argument('--place',action='store_true');p.add_argument('--place-duration',type=float,default=.35);p.add_argument('--rotate',type=float,default=0.);p.add_argument('--inward',type=float,default=.01);run(**vars(p.parse_args()))
