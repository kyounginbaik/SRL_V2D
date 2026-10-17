"""Distinguish a short opposing-contact event from retaining and lifting an object."""
from pathlib import Path
import argparse,json
import numpy as np,mujoco


def audit(execution_dir):
    directory=Path(execution_dir)
    events=json.loads((directory/'acquisition_events.json').read_text())
    config=events['config'];samples=events['samples'];name=config['object_name']
    capture=next((e['time_seconds'] for e in events['events'] if e['phase']=='dwell'),None)
    transport=next((e['time_seconds'] for e in events['events'] if e['phase']=='transport'),None)
    data=np.load(directory/'trajectory.npz');names=list(data['object_names'].astype(str));oid=names.index(name)
    times=data['timestamps_seconds'];positions=data['object_positions'][:,oid]
    start=config['acquisition_start_s'];stop=min(float(times[-1]),4.7)
    z0=float(positions[np.argmin(abs(times-start)),2]);maxlift=float(np.max(positions[(times>=start)&(times<=stop),2])-z0)
    contact_times=np.array([row[0] for row in samples]);both=np.array([row[3] and row[4] for row in samples])
    relevant=(contact_times>=start)&(contact_times<=stop)
    dt=float(np.median(np.diff(contact_times)));longest=0;count=0
    for flag in both[relevant]:
        count=count+1 if flag else 0;longest=max(longest,count)
    model=mujoco.MjModel.from_xml_path(str(directory/'scene_physics.xml'));state=mujoco.MjData(model)
    obj=model.body(name).id;table=model.body('table').id;table_frames=[]
    for index,t in enumerate(times):
        if not(start<=t<=stop):continue
        state.qpos[:]=data['qpos'][index];mujoco.mj_fwdPosition(model,state)
        supported=any(set((int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])))=={obj,table} for c in state.contact)
        table_frames.append(dict(time_s=float(t),table_contact=bool(supported),height_change_m=float(positions[index,2]-z0)))
    airborne=[r for r in table_frames if not r['table_contact'] and r['height_change_m']>.01]
    result=dict(object_name=name,contact_capture_time_s=capture,transport_time_s=transport,
        contact_capture_is_not_task_success=True,examined_interval_s=[start,stop],
        opposing_contact_total_s=float(np.sum(both[relevant])*dt),
        longest_opposing_contact_s=float(longest*dt),
        max_object_height_gain_m=maxlift,
        observed_airborne_above_1cm_frames=len(airborne),
        observed_airborne_above_1cm_seconds=float(len(airborne)*np.median(np.diff(times))),
        table_contact_and_height_samples=table_frames)
    if transport is not None:
        window=(contact_times>=transport)&(contact_times<transport+.5)
        result['opposing_contact_fraction_first_half_second_transport']=float(both[window].mean()) if window.any() else None
    (directory/'retention_audit.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('execution_dir',type=Path)
    result=audit(parser.parse_args().execution_dir)
    print(json.dumps({k:v for k,v in result.items() if k!='table_contact_and_height_samples'},indent=2))
