"""Clear an already released hand before it recontacts a supported spoon."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
from physics_pickup_search import B,H,TR,blend,dump,execute_reference,evaluate_files
from audit_object_support import audit as audit_support
from audit_full_joint_limits import main as audit_joints

def main(start):
    source=B/'refinement_20261018/episode_000036/right_only_upright/variant_01'
    out=source.parent.parent/'right_only_release'/f'start_{start:.2f}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    previous=json.loads((source/'result.json').read_text());scene=Path(previous['execution']['scene']);m=mujoco.MjModel.from_xml_path(str(scene))
    ref=TR.load_reference(source/'reference.npz',m);Q=ref['qpos'].copy();old=Q.copy();t=np.arange(len(Q))/ref['fps'];k=round(start*ref['fps'])
    z=np.load(source/'execution/trajectory.npz');d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['initial_state'],int(z['state_spec']));n=round(start/m.opt.timestep)
    for ctrl in z['substep_ctrl'][:n]:d.ctrl[:]=ctrl;mujoco.mj_step(m,d)
    assert np.array_equal(d.qpos,z['substep_qpos'][n-1])
    rig=H.HandRig.build(m,'robot_right_','right');c=np.array([rig.wrist_qpos[key] for key in ['x','y','z','rx','ry','rz']]+list(rig.finger_qpos.values()));ob=m.body('wooden_spoon').id
    force=np.zeros(6);hand=0.;support=0.
    for i,contact in enumerate(d.contact):
        b1,b2=int(m.geom_bodyid[contact.geom1]),int(m.geom_bodyid[contact.geom2]);other=b2 if b1==ob else b1 if b2==ob else -1
        if other<0:continue
        mujoco.mj_contactForce(m,d,i,force);fn=max(0.,float(force[0]))
        if m.body(other).name.startswith('robot_'):hand+=fn
        else:support+=fn
    if support<.1 or hand>.1:raise ValueError(f'Anchor is not already released and physically supported: {support=}, {hand=}')
    direction=d.qpos[c[:3]]-d.xpos[ob];direction[2]=0.;direction/=np.linalg.norm(direction)
    retreat=.14*direction+np.array([0.,0.,.10]);mask=t>start;alpha=blend((t[mask]-start)/.7)
    Q[np.ix_(mask,c)]=Q[k,c];Q[np.ix_(mask,c[:3])]+=alpha[:,None]*retreat
    assert np.array_equal(Q[:k+1],old[:k+1]);other=np.setdiff1d(np.arange(m.nq),c);assert np.array_equal(Q[:,other],old[:,other])
    reference=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source='Already-released, table-supported spoon: hold open posture and withdraw hand clear; all original object goals/timing unchanged; alternate right-only carry',floating_runtime=True)
    inp=B/'episodes/episode_000036';manifest=json.loads((inp/'input_manifest.json').read_text())
    execution=execute_reference(scene,reference,out/'execution');metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    with np.load(execution['trajectory']) as new:assert np.array_equal(new['substep_qpos'][:n],z['substep_qpos'][:n])
    dump(out/'result.json',dict(episode='episode_000036',input_dir=str(inp),strategy='right_only_alternative_not_handover',source_reference=str(source/'reference.npz'),protected_prefix_s=start,initial_support_n=support,initial_hand_force_n=hand,withdrawal_m=retreat,reference=str(reference),execution=execution,metrics=metrics,learned_policy=False))
    audit_support(out);audit_joints(out);print(json.dumps(dict(start=start,objects=metrics['declared_frame_reference_proxy']['objects'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=float,required=True);main(p.parse_args().start)
