#!/usr/bin/env python3
"""Test small unused-joint command margins without changing physical limits."""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from physics_pickup_search_v2 import TR,blend,dump
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from audit_joint_limits import audit


def run(source,output,degrees,side):
    r=json.loads((source/'result.json').read_text());ex=r['execution'];m=mujoco.MjModel.from_xml_path(ex['scene'])
    ref=TR.load_reference(r['reference'],m);Q=ref['qpos'].copy();j=m.joint(f'robot_{side}_{side}_pinky_CMC').id
    q=int(m.jnt_qposadr[j]);a=int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]);t=np.arange(len(Q))/ref['fps']
    Q[:,q]=np.clip(Q[:,q]+np.radians(degrees)*blend((t-.1)/.3),*m.actuator_ctrlrange[a])
    output.mkdir(parents=True,exist_ok=True)
    path=TR.save_reference(output/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],
        source='Bounded unused pinky CMC target margin; exact source object references and physical joint limits',floating_runtime=True)
    execution=execute_reference(ex['scene'],path,output/'execution');s=r['spec']
    metrics=evaluate_files(s['source'],execution['trajectory'],s['manifest'],execution['scene'],s['geometry'])
    quality=audit(execution['scene'],execution['trajectory']);dump(output/'audit/actual_joint_limits.json',quality)
    result=dict(episode=s.get('episode'),reference=str(path),execution=execution,metrics=metrics,spec=s,
        parent=str(source),command_margin_degrees=degrees,command_side=side,physical_limits_changed=False)
    dump(output/'result.json',result)
    print(json.dumps(dict(output=str(output),mp_sr=metrics['reference_proxy_mp_sr'],
        objects=metrics['official_style_reference_proxy']['objects'],max_actual_joint_excursion_degrees=quality['hinge_max_violation_degrees'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--degrees',type=float,required=True);p.add_argument('--side',choices=['left','right'],default='right');run(**vars(p.parse_args()))
