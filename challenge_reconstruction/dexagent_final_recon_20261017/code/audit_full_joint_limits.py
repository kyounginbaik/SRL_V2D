"""Measure both hands' actual soft-limit excursions over every physics substep."""
from pathlib import Path
import argparse,json
import mujoco,numpy as np


def main(run):
    run=Path(run).resolve();m=mujoco.MjModel.from_xml_path(str(run/'execution/scene_physics.xml'))
    with np.load(run/'execution/trajectory.npz',allow_pickle=False) as z:
        q=z['substep_qpos'];ctrl=z['substep_ctrl'];time=z['substep_time_s']
    rows={}
    for j in range(m.njnt):
        name=m.joint(j).name
        if not m.jnt_limited[j] or not name.startswith(('robot_left_','robot_right_')):continue
        a=int(m.jnt_qposadr[j]);lo,hi=m.jnt_range[j];viol=np.maximum(np.maximum(lo-q[:,a],q[:,a]-hi),0);k=int(viol.argmax())
        aids=np.flatnonzero(m.actuator_trnid[:,0]==j);aid=int(aids[0])
        angular=m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE
        rows[name]=dict(type='hinge' if angular else 'slide',range=m.jnt_range[j].tolist(),max_actual_violation=float(viol[k]),max_actual_violation_deg=float(np.degrees(viol[k])) if angular else None,
            maximum_at_s=float(time[k]),actual_at_max=float(q[k,a]),command_at_max=float(ctrl[k,aid]),
            max_command_violation=float(np.maximum(np.maximum(lo-ctrl[:,aid],ctrl[:,aid]-hi),0).max()))
    maxima={side:max(v['max_actual_violation_deg'] for n,v in rows.items() if n.startswith(f'robot_{side}_') and v['type']=='hinge') for side in ['left','right']}
    report=dict(full_physics_substeps=len(q),horizon_s=[float(time[0]),float(time[-1])],both_hands_audited=True,maximum_angular_violation_deg=maxima,joints=rows,
        interpretation='MuJoCo joint limits are soft constraints. Actual state excursions are reported separately from bounded servo target commands. Independent replay and object tracking alone do not establish anatomical quality.')
    out=run/'audit';out.mkdir(exist_ok=True);(out/'full_joint_limits.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(run=str(run),maximum_angular_violation_deg=maxima)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(**vars(p.parse_args()))
