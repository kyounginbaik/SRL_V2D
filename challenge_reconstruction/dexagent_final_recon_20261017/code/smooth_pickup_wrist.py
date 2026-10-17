#!/usr/bin/env python3
"""Test bounded reference smoothing around Euler wrist singularities.

Only robot targets change after a preserved pickup prefix. The physical model,
all object targets and the original episode clock remain identical.
"""
from pathlib import Path
import argparse,json
import numpy as np,mujoco
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from physics_pickup_search_v2 import TR,blend,dump
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files
from audit_joint_limits import audit


def run(source,output,sigma,start_s):
    r=json.loads((source/'result.json').read_text());spec=r['spec'];ex=r['execution'];m=mujoco.MjModel.from_xml_path(ex['scene'])
    ref=TR.load_reference(r['reference'],m);Q=ref['qpos'].copy();fps=ref['fps'];t=np.arange(len(Q))/fps
    cols=np.array([int(m.joint('robot_right_right_rot_'+a).qposadr[0]) for a in 'xyz'])
    before=Q[:,cols].copy();target=gaussian_filter1d(before,sigma,axis=0,mode='nearest')
    weight=blend((t-start_s)/.5);Q[:,cols]=before+weight[:,None]*(target-before)
    assert np.array_equal(Q[t<=start_s],ref['qpos'][t<=start_s])
    delta=np.degrees((Rotation.from_euler('XYZ',before).inv()*Rotation.from_euler('XYZ',Q[:,cols])).magnitude())
    output.mkdir(parents=True,exist_ok=True)
    path=TR.save_reference(output/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],
        source='Robot wrist reference smoothing after preserved pickup; physical limits and immutable source targets unchanged',floating_runtime=True)
    execution=execute_reference(ex['scene'],path,output/'execution')
    metrics=evaluate_files(spec['source'],execution['trajectory'],spec['manifest'],execution['scene'],spec['geometry'])
    quality=audit(execution['scene'],execution['trajectory']);dump(output/'audit/actual_joint_limits.json',quality)
    result=dict(episode=spec.get('episode'),reference=str(path),execution=execution,metrics=metrics,spec=spec,parent=str(source),
        sigma_frames=sigma,preserved_until_s=start_s,robot_target_rotation_change_max_degrees=float(delta.max()),
        robot_target_rotation_change_mean_degrees=float(delta.mean()),
        max_target_euler_speed_before_rad_s=float(np.abs(np.diff(before,axis=0)).max()*fps),
        max_target_euler_speed_after_rad_s=float(np.abs(np.diff(Q[:,cols],axis=0)).max()*fps),
        physical_limits_changed=False,object_targets_changed=False)
    dump(output/'result.json',result)
    print(json.dumps(dict(output=str(output),mp_sr=metrics['reference_proxy_mp_sr'],objects=metrics['official_style_reference_proxy']['objects'],
        rotation_change_max_degrees=float(delta.max()))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sigma',type=float,required=True);p.add_argument('--start-s',type=float,required=True);run(**vars(p.parse_args()))
