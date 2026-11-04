"""Evaluate bounded command margins against actual physical soft-joint limits.

Changing a contact finger may change the grasp: every trial is freshly executed
and scored over the complete original object timeline. Model limits stay fixed.
"""
from pathlib import Path
import argparse,json
import mujoco,numpy as np
from physics_pickup_search_v2 import TR,blend,dump
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from audit_joint_limits import audit

def run(source,episode_dir,output,margins):
    old=json.loads((source/'result.json').read_text());ex=old['execution'];m=mujoco.MjModel.from_xml_path(ex['scene'])
    ref=TR.load_reference(old['reference'],m);Q=ref['qpos'].copy();t=np.arange(len(Q))/ref['fps']
    changed=[]
    for name,degrees in margins.items():
        j=m.joint(name).id;q=int(m.jnt_qposadr[j]);a=int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]);changed.append(q)
        Q[:,q]=np.clip(Q[:,q]+np.radians(degrees)*blend((t-.1)/.3),*m.actuator_ctrlrange[a])
    unchanged=np.setdiff1d(np.arange(m.nq),changed)
    assert np.array_equal(Q[:,unchanged],ref['qpos'][:,unchanged]);assert np.array_equal(Q[0],ref['qpos'][0])
    output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():raise FileExistsError(output)
    path=TR.save_reference(output/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],
        source='Bounded finger command margins; actual physical joint limits, objects and original timing unchanged',floating_runtime=True)
    execution=execute_reference(ex['scene'],path,output/'execution');manifest=json.loads((episode_dir/'input_manifest.json').read_text())
    spec=dict(source=manifest['source'],manifest=str(episode_dir/'input_manifest.json'),geometry=str(episode_dir/'metric_geometry.json'),episode=manifest['episode'])
    metrics=evaluate_files(spec['source'],execution['trajectory'],spec['manifest'],execution['scene'],spec['geometry'])
    quality=audit(execution['scene'],execution['trajectory']);dump(output/'audit/actual_joint_limits.json',quality)
    result=dict(episode=manifest['episode'],reference=str(path),execution=execution,metrics=metrics,spec=spec,parent=str(source),
        command_margins_degrees=margins,physical_limits_changed=False,learned_policy=False)
    dump(output/'result.json',result)
    print(json.dumps(dict(output=str(output),mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['declared_frame_reference_proxy']['objects'],
        max_actual_joint_excursion_degrees=quality['hinge_max_violation_degrees'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['source','episode-dir','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--margins',type=json.loads,required=True);run(**vars(p.parse_args()))
