"""Test collision-free robot alternatives to an infeasible below-table wrist.

Object reference poses remain bytewise unchanged. Any transient mismatch is
fully included in MP-SR. This does not claim reproduction of human handovers.
"""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
from physics_pickup_search import B,H,TR,blend,dump,execute_reference,evaluate_files
from wrist_reference import continuous_euler_xyz

def main(variant):
    source=B/'refinement_20261018/episode_000036/right_only_fallback/variant_01'
    out=source.parent.parent/'right_only_upright'/f'variant_{variant:02d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out)
    previous=json.loads((source/'result.json').read_text());scene=Path(previous['execution']['scene']);m=mujoco.MjModel.from_xml_path(str(scene))
    ref=TR.load_reference(source/'reference.npz',m);Q=ref['qpos'].copy();old=Q.copy();t=np.arange(len(Q))/ref['fps']
    rig=H.HandRig.build(m,'robot_right_','right');c=np.array([rig.wrist_qpos[key] for key in ['x','y','z','rx','ry','rz']])
    if variant<2:
        begin,end=[(14.65,15.55),(14.4,15.8)][variant];i,j=round(begin*ref['fps']),round(end*ref['fps'])
        alpha=blend((t[i:j+1]-begin)/(end-begin));Q[i:j+1,c[:3]]=(1-alpha[:,None])*old[i,c[:3]]+alpha[:,None]*old[j,c[:3]]
        a=Rotation.from_euler('XYZ',old[i,c[3:]]);b=Rotation.from_euler('XYZ',old[j,c[3:]])
        rr=a*Rotation.from_rotvec(alpha[:,None]*(a.inv()*b).as_rotvec())
        Q[i:j+1,c[3:]]=continuous_euler_xyz(rr,old[i,c[3:]])
        Q[:i+1]=old[:i+1];Q[j]=old[j]
        note='Smooth robot wrist bridge avoids the infeasible upright-grasp configuration; unchanged target mismatch is scored.'
    else:
        begin=14.35;end=15.8
        dz=.20*blend((t-begin)/.5)*(1-blend((t-15.3)/.5));Q[:,c[2]]+=dz
        note='Temporary20cm robot hand lift avoids table blockage through original rotation; unchanged target mismatch is scored.'
    assert np.array_equal(Q[t<=begin],old[t<=begin]);other=np.setdiff1d(np.arange(m.nq),c);assert np.array_equal(Q[:,other],old[:,other])
    reference=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source=note,floating_runtime=True)
    inp=B/'episodes/episode_000036';manifest=json.loads((inp/'input_manifest.json').read_text())
    execution=execute_reference(scene,reference,out/'execution');metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    dump(out/'result.json',dict(episode='episode_000036',input_dir=str(inp),strategy='right_only_alternative_not_handover',source_reference=str(source/'reference.npz'),note=note,window=[begin,end],reference=str(reference),execution=execution,metrics=metrics,learned_policy=False))
    print(json.dumps(dict(variant=variant,objects=metrics['declared_frame_reference_proxy']['objects'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',type=int,choices=range(3),required=True);main(p.parse_args().variant)
