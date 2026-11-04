"""Resolve physical force paths when a lid pickup also moves the pot."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco
from physics_pickup_search import PickupExperiment,dump

B=Path(__file__).resolve().parents[1]


def audit(search_dir):
    path=Path(search_dir);settings=json.loads((path/'settings.json').read_text());spec=settings['spec']
    result=json.loads((path/'result.json').read_text());exp=PickupExperiment(spec)
    Q,_=exp.commands(result['parameters']);m=exp.m;d=mujoco.MjData(m);mujoco.mj_copyData(d,m,exp.prefix)
    force=np.zeros(6);rows=[]
    for step in range(exp.start_step,exp.end_step):
        d.ctrl[:]=exp.control(Q,step);mujoco.mj_step(m,d)
        if step%5!=4:continue
        totals={};pairs=[]
        for i,c in enumerate(d.contact):
            b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);n1,n2=m.body(b1).name,m.body(b2).name
            if not ('white_pot' in (n1,n2) or 'white_pot_lid' in (n1,n2)):continue
            mujoco.mj_contactForce(m,d,i,force)
            if force[0]<.02:continue
            obj=n1 if n1 in ('white_pot','white_pot_lid') else n2;other=n2 if obj==n1 else n1
            if other.startswith('robot_'):
                side='left' if other.startswith('robot_left_') else 'right'
                finger=next((f for f in ('thumb','index','middle','ring','pinky') if '_'+f+'_' in other),'palm')
                key=f'{side}_{finger}--{obj}'
            else:key='--'.join(sorted((obj,other)))
            totals[key]=totals.get(key,0.)+float(force[0])
            pairs.append(dict(key=key,geom1=m.geom(c.geom1).name,geom2=m.geom(c.geom2).name,force_n=float(force[0]),penetration_m=max(0.,-float(c.dist))))
        rows.append(dict(time_seconds=float(d.time),forces_n=totals,pairs=pairs,
            object_positions={n:d.qpos[a:a+3].tolist() for n,a in exp.objects.items()}))
    summary=[]
    for sec in (1.5,1.8,2.,2.2,2.4,2.6,2.8,3.,3.5,4.):
        row=min(rows,key=lambda r:abs(r['time_seconds']-sec));summary.append({k:v for k,v in row.items() if k!='pairs'})
    output=B/'refinement_20261018/episode_000001/coupling_audit.json'
    dump(output,dict(source=str(path),summary=summary,samples=rows));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--search-dir',default=str(B/'refinement_20261018/episode_000001/pickup_cem_v1/pose_01'))
    audit(p.parse_args().search_dir)
