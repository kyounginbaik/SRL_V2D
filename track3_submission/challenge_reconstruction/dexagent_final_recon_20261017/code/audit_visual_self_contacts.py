#!/usr/bin/env python3
"""Replay recorded controls and quantify actual explicit visual hand self-contacts.

No scene edits. This detects active forces, not whether the collision design is
appropriate. Reported contact forces are solver forces from exact control replay.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import mujoco


def audit(scene, trajectory, output):
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    with np.load(trajectory,allow_pickle=False) as z:
        ctrl=z['substep_ctrl'];q=z['substep_qpos'];v=z['substep_qvel'];times=z['substep_time_s']
        mujoco.mj_setState(model,data,z['initial_state'],mujoco.mjtState(int(z['state_spec'])))
    mujoco.mj_forward(model,data)
    names=[model.geom(i).name or '' for i in range(model.ngeom)]
    bodies=[model.body(int(x)).name or '' for x in model.geom_bodyid]
    visual={i for i,n in enumerate(names) if 'visual' in n and bodies[i].startswith('robot_')}
    masked={i for i in visual if model.geom_contype[i]==0 and model.geom_conaffinity[i]==0}
    explicit=[p for p in range(model.npair) if int(model.pair_geom1[p]) in masked or int(model.pair_geom2[p]) in masked]
    touched={};force=np.zeros(6);maxq=maxv=0.;active_steps=0;peak_self_generalized=np.zeros(model.nv)
    for step,u in enumerate(ctrl):
        data.ctrl[:]=u;mujoco.mj_step(model,data)
        maxq=max(maxq,float(np.max(np.abs(q[step]-data.qpos))));maxv=max(maxv,float(np.max(np.abs(v[step]-data.qvel))))
        active=False;generalized=np.zeros(model.nv)
        for ci in range(data.ncon):
            c=data.contact[ci];g1,g2=int(c.geom1),int(c.geom2)
            if not (bodies[g1].startswith('robot_') and bodies[g2].startswith('robot_')):
                continue
            if g1 not in masked and g2 not in masked:
                continue
            mujoco.mj_contactForce(model,data,ci,force)
            magnitude=float(np.linalg.norm(force[:3]));active |= magnitude>1e-9
            pair=' | '.join(sorted([names[g1],names[g2]]))
            r=touched.setdefault(pair,dict(contact_samples=0,force_samples=0,max_force_n=0.,max_penetration_m=0.,first_time_s=float(times[step]),last_time_s=0.))
            r['contact_samples']+=1;r['force_samples']+=int(magnitude>1e-9);r['last_time_s']=float(times[step])
            r['max_force_n']=max(r['max_force_n'],magnitude);r['max_penetration_m']=max(r['max_penetration_m'],max(0.,-float(c.dist)))
            frame=c.frame.reshape(3,3);fw=frame.T@force[:3];tw=frame.T@force[3:]
            mujoco.mj_applyFT(model,data,-fw,-tw,c.pos,int(model.geom_bodyid[g1]),generalized)
            mujoco.mj_applyFT(model,data,fw,tw,c.pos,int(model.geom_bodyid[g2]),generalized)
        active_steps+=int(active);peak_self_generalized=np.maximum(peak_self_generalized,np.abs(generalized))
    rows=sorted([dict(pair=p,**r) for p,r in touched.items()],key=lambda r:r['max_force_n'],reverse=True)
    result=dict(scene=str(scene),trajectory=str(trajectory),samples=len(ctrl),dt=float(model.opt.timestep),
        scene_sha256=hashlib.sha256(Path(scene).read_bytes()).hexdigest(),
        recording_sha256=hashlib.sha256(Path(trajectory).read_bytes()).hexdigest(),
        exact_control_replay=maxq==0 and maxv==0,max_qpos_error=maxq,max_qvel_error=maxv,
        masked_visual_geoms=len(masked),explicit_pairs_using_masked_visual=len(explicit),
        steps_with_force_from_masked_visual_self_contacts=active_steps,
        fraction_of_steps_with_masked_visual_self_forces=active_steps/len(ctrl),
        peak_generalized_self_contact_load=peak_self_generalized.tolist(),
        contacted_pairs=rows,
        scope='Actual solver forces during independent exact control replay. Geometry with collision masks zero can still collide via explicit pairs. This demonstrates activity, not whether the intended self-collision design is correct.')
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['exact_control_replay','masked_visual_geoms','explicit_pairs_using_masked_visual','fraction_of_steps_with_masked_visual_self_forces']}))
    print(json.dumps(rows[:5],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    audit(**vars(p.parse_args()))
