#!/usr/bin/env python3
"""Report actual soft-joint excursions separately from bounded target commands."""
from pathlib import Path
import argparse
import json
import numpy as np
import mujoco


def audit(scene, trajectory):
    model=mujoco.MjModel.from_xml_path(str(scene))
    with np.load(trajectory) as recording:
        key='substep_qpos' if 'substep_qpos' in recording else 'qpos'
        q=np.asarray(recording[key]);timekey='substep_time_s' if key.startswith('substep') else 'time_s'
        times=np.asarray(recording[timekey]);rows=[]
        for joint in range(model.njnt):
            if not model.jnt_limited[joint]:continue
            kind=int(model.jnt_type[joint])
            if kind not in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE)):continue
            x=q[:,model.jnt_qposadr[joint]];lo,hi=model.jnt_range[joint]
            excess=np.maximum(np.maximum(lo-x,x-hi),0.)
            unit='degrees' if kind==int(mujoco.mjtJoint.mjJNT_HINGE) else 'millimeters'
            scale=180./np.pi if unit=='degrees' else 1000.
            peak=int(excess.argmax())
            rows.append(dict(joint=model.joint(joint).name,unit=unit,
                max_violation=float(excess[peak]*scale),mean_violation=float(excess.mean()*scale),
                fraction_outside=float(np.mean(excess>1e-7)),peak_time_seconds=float(times[peak]),
                lower_limit=float(lo*scale),upper_limit=float(hi*scale),peak_actual_value=float(x[peak]*scale)))
        rows.sort(key=lambda r:r['max_violation'],reverse=True)
    return dict(scene=str(Path(scene).resolve()),trajectory=str(Path(trajectory).resolve()),
        sampling=key,samples=len(q),joints=rows,
        interpretation='Physical soft limits may be exceeded even when servo commands are bounded. This is a quality diagnostic, separate from official MP-SR; no organizer tolerance is inferred.',
        hinge_max_violation_degrees=max((x['max_violation'] for x in rows if x['unit']=='degrees'),default=0.))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.scene,a.trajectory);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output),worst=result['joints'][:3]),indent=2))
