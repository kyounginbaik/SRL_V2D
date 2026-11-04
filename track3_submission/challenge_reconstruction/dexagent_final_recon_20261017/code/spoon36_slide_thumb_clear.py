"""Slide the thumb sideways clear of the spoon before lifting the hand.

The direct upward withdrawal hooks the underside of the thin handle. This
uses its measured supported state to choose a planar transverse escape first.
"""
import argparse,json
import numpy as np,mujoco
from physics_pickup_search import B,H,TR,blend,dump,execute_reference,evaluate_files
from audit_object_support import audit as audit_support
from audit_full_joint_limits import main as audit_joints

def main(distance):
    source=B/'refinement_20261018/episode_000036/right_only_thumb_release/open_0.016'
    out=source.parent.parent/'right_only_transverse_release'/f'slide_{distance:.3f}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    scene=source/'execution/scene_physics.xml';m=mujoco.MjModel.from_xml_path(str(scene));ref=TR.load_reference(source/'reference.npz',m);Q=ref['qpos'].copy();old=Q.copy();t=np.arange(len(Q))/ref['fps'];start=18.8;k=round(start*ref['fps'])
    z=np.load(source/'execution/trajectory.npz');d=mujoco.MjData(m);d.qpos[:]=z['qpos'][k];mujoco.mj_forward(m,d)
    rig=H.HandRig.build(m,'robot_right_','right');c=np.array([rig.wrist_qpos[key] for key in ['x','y','z','rx','ry','rz']]);ob=m.body('wooden_spoon').id;tip=d.site_xpos[rig.fingertip_sites['thumb']]
    line=d.xpos[ob]-tip;line[2]=0;line/=np.linalg.norm(line);side=np.array([-line[1],line[0],0.])
    palm=d.qpos[c[:3]]-d.xpos[ob]
    if side@palm<0:side=-side
    retreat=palm.copy();retreat[2]=0;retreat/=np.linalg.norm(retreat);retreat=.14*retreat+np.array([0.,0.,.12])
    mask=t>start;a=blend((t[mask]-start)/.35);b=blend((t[mask]-19.15)/.75)
    Q[np.ix_(mask,c[:3])]=old[k,c[:3]]+a[:,None]*distance*side+b[:,None]*retreat
    assert np.array_equal(Q[:k+1],old[:k+1]);other=np.setdiff1d(np.arange(m.nq),c[:3]);assert np.array_equal(Q[:,other],old[:,other])
    reference=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source='Robot thumb transverse clearance before upward retreat from supported spoon; original object goals and exact physical prefix preserved',floating_runtime=True)
    inp=B/'episodes/episode_000036';manifest=json.loads((inp/'input_manifest.json').read_text());execution=execute_reference(scene,reference,out/'execution');metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    n=round(start/m.opt.timestep)
    with np.load(execution['trajectory']) as new:assert np.array_equal(new['substep_qpos'][:n],z['substep_qpos'][:n])
    dump(out/'result.json',dict(episode='episode_000036',input_dir=str(inp),strategy='right_only_alternative_not_handover',source_reference=str(source/'reference.npz'),protected_prefix_s=start,transverse_clearance_m=distance*side,withdrawal_m=retreat,reference=str(reference),execution=execution,metrics=metrics,learned_policy=False))
    audit_support(out);audit_joints(out);print(json.dumps(dict(distance=distance,objects=metrics['declared_frame_reference_proxy']['objects'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--distance',type=float,required=True);main(p.parse_args().distance)
