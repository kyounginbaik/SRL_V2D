"""Measure release support and hand/table contact by exact control replay.

Diagnostics only: this does not modify simulation state, targets or scores.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import mujoco


def run(run, object_name, start_s, final_window_s=.5):
    folder=Path(run);report=json.loads((folder/'result.json').read_text())
    ex=report['execution'];m=mujoco.MjModel.from_xml_path(ex['scene']);d=mujoco.MjData(m)
    with np.load(ex['trajectory'],allow_pickle=False) as z:
        controls=z['substep_ctrl'].copy();initial=z['initial_state'].copy()
        state_spec=int(z['state_spec']);expected_q=z['substep_qpos'][-1].copy()
        expected_v=z['substep_qvel'][-1].copy()
    mujoco.mj_setState(m,d,initial,state_spec)
    ob=m.body(object_name).id;table=m.body('table').id
    dof=int(m.body_dofadr[ob]);robot={b for b in range(m.nbody) if m.body(b).name.startswith('robot_')}
    rows=[];force=np.zeros(6);peaks={};initial_height=None
    for ctrl in controls:
        d.ctrl[:]=ctrl;mujoco.mj_step(m,d)
        if d.time+1e-12<start_s:continue
        if initial_height is None:initial_height=float(d.xpos[ob,2])
        hand=0.;support=0.;hand_table=0.
        for i,c in enumerate(d.contact):
            a,b=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])
            if ob not in (a,b) and not ((a in robot and b==table) or (b in robot and a==table)):continue
            mujoco.mj_contactForce(m,d,i,force);fn=max(0.,float(force[0]))
            if ob in (a,b):
                other=b if a==ob else a
                if other in robot:hand+=fn
                else:support+=fn
            else:
                hand_table+=fn;name=m.body(a if a in robot else b).name
                if fn>peaks.get(name,{}).get('force_n',0.):
                    peaks[name]=dict(force_n=fn,time_s=float(d.time),penetration_m=max(0.,-float(c.dist)))
        rows.append([d.time,hand,support,hand_table,d.xpos[ob,2],d.qvel[dof+2],np.linalg.norm(d.qvel[dof+3:dof+6])])
    qerror=float(np.max(abs(d.qpos-expected_q)));verror=float(np.max(abs(d.qvel-expected_v)))
    if qerror!=0. or verror!=0.:raise RuntimeError('Diagnostic replay changed the physical execution')
    a=np.asarray(rows);end=a[:,0]>=d.time-final_window_s;touch=a[:,1]>.01
    result=dict(object=object_name,start_s=start_s,final_window_s=final_window_s,
        full_control_replay_qpos_max_error=qerror,full_control_replay_qvel_max_error=verror,
        final_nonhand_support_fraction=float(np.mean(a[end,2]>.1)),
        final_hand_free_fraction=float(np.mean(a[end,1]<.01)),
        final_support_force_mean_n=float(np.mean(a[end,2])),
        last_hand_contact_s=float(a[touch,0][-1]) if np.any(touch) else None,
        max_object_rise_after_start_mm=float(1000*max(0.,a[:,4].max()-initial_height)),
        max_upward_velocity_m_s=float(max(0.,a[:,5].max())),
        max_angular_speed_rad_s=float(a[:,6].max()),
        max_summed_hand_table_force_n=float(a[:,3].max()),
        hand_table_force_above_5n_duration_s=float(np.sum(a[:,3]>5)*m.opt.timestep),
        hand_table_contact_peaks=peaks,
        interpretation='Force thresholds describe this audit only; they are not official competition acceptance thresholds.')
    out=folder/'audit';out.mkdir(exist_ok=True)
    np.savez_compressed(out/f'{object_name}_release_timeseries.npz',values=a,
        columns=np.array(['time_s','hand_force_n','support_force_n','hand_table_force_n','height_m','vertical_speed_m_s','angular_speed_rad_s']))
    (out/f'{object_name}_release_support.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    p.add_argument('--object-name',required=True);p.add_argument('--start-s',type=float,required=True)
    p.add_argument('--final-window-s',type=float,default=.5);run(**vars(p.parse_args()))
