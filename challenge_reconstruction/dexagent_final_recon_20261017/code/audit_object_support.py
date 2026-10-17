"""Replay exact controls and report force-bearing robot/nonrobot support."""
from pathlib import Path
import argparse,json
import mujoco,numpy as np

def audit(run):
    run=Path(run);scene=run/'execution/scene_physics.xml';path=run/'execution/trajectory.npz'
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
    with np.load(path,allow_pickle=False) as z:
        controls=z['substep_ctrl'];states=z['substep_qpos'];initial=z['initial_state'];spec=int(z['state_spec']);names=z['object_names'].tolist()
    mujoco.mj_setState(m,d,initial,spec);bids={m.body(name).id:name for name in names};force=np.zeros(6);rows=[]
    for step,ctrl in enumerate(controls):
        d.ctrl[:]=ctrl;mujoco.mj_step(m,d)
        if step%25!=24:continue
        values={name:dict(support={},fingers={},origin_m=d.xpos[bid].tolist()) for bid,name in bids.items()}
        for i,c in enumerate(d.contact):
            b1,b2=int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2]);mujoco.mj_contactForce(m,d,i,force);fn=max(0.,float(force[0]))
            if fn<.001:continue
            for ob,other in [(b1,b2),(b2,b1)]:
                if ob not in bids:continue
                name=m.body(other).name;key='fingers' if name.startswith('robot_') else 'support'
                bucket=values[bids[ob]][key];bucket[name]=bucket.get(name,0.)+fn
        rows.append(dict(time_s=float(d.time),objects=values))
    err=float(abs(d.qpos-states[-1]).max());assert err==0.
    out=run/'audit/object_support.json';out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(dict(exact_replay_max_final_qpos_error=err,rows=rows),indent=2)+'\n')
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);audit(p.parse_args().run)
