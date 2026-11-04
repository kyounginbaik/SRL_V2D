"""Physically supported end release; full recording/replay remains mandatory."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from repair_episode1_registration import OUT
from physics_pickup_search_gravity import H,TR,dump,blend,execute_reference,evaluate_files
from supported_release import plan_supported_release

def run(base=None,start_s=15.15,left_clearance=False):
    root=OUT/'patch_height094';base=Path(base or root/'lid_seat_07.0mm');out=root/('hands_release_v2' if left_clearance else 'hands_release_v1');out.mkdir(exist_ok=True)
    r=json.loads((base/'result.json').read_text());scene=r['execution']['scene'];record=r['execution']['trajectory'];spec=r['spec']
    m=mujoco.MjModel.from_xml_path(scene);ref=TR.load_reference(base/'reference.npz',m);z=np.load(record);fps=ref['fps'];idx=round(start_s*fps)
    d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['initial_state'],int(z['state_spec']));count=round(start_s/m.opt.timestep)
    for ctrl in z['substep_ctrl'][:count]:d.ctrl[:]=ctrl;mujoco.mj_step(m,d)
    assert np.array_equal(d.qpos,z['substep_qpos'][count-1]);force=np.zeros(6);contacts={};leftforce=0.;pot_table=0.;lid_pot=0.
    for i,c in enumerate(d.contact):
        a,b=m.body(m.geom_bodyid[c.geom1]).name,m.body(m.geom_bodyid[c.geom2]).name
        if not any(v in ('white_pot','white_pot_lid') for v in (a,b)):continue
        mujoco.mj_contactForce(m,d,i,force);fn=max(0.,float(force[0]));key=' / '.join(sorted((a,b)));contacts[key]=contacts.get(key,0.)+fn
        if {a,b}=={'table','white_pot'}:pot_table+=fn
        if {a,b}=={'white_pot','white_pot_lid'}:lid_pot+=fn
        if 'white_pot' in (a,b) and any(v.startswith('robot_left_') for v in (a,b)):leftforce+=fn
    dump(out/'support_anchor.json',dict(start_s=start_s,contacts=contacts,pot_table_force_n=pot_table,lid_pot_force_n=lid_pot,left_pot_force_n=leftforce))
    if pot_table<.5 or lid_pot<.1:raise ValueError('Release requires lid→pot→table physical support path')
    Q,meta=plan_supported_release(scene,record,base/'reference.npz',side='right',object_name='white_pot_lid',start_s=start_s,
        open_complete_s=start_s+.30,clear_s=16.3,opening_m=.008,withdrawal_horizontal_m=.10,withdrawal_up_m=.10)
    rig=H.HandRig.build(m,'robot_left_','left');wc=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')]);fc=np.array(list(rig.finger_qpos.values()));up=-m.opt.gravity/np.linalg.norm(m.opt.gravity)
    if leftforce>.1:
        L,lmeta=plan_supported_release(scene,record,base/'reference.npz',side='left',object_name='white_pot',start_s=start_s,
            open_complete_s=start_s+.30,clear_s=16.3,opening_m=.008,withdrawal_horizontal_m=.10,withdrawal_up_m=.10)
        Q[:,np.r_[wc,fc]]=L[:,np.r_[wc,fc]]
    else:
        # A hand with no force-bearing grasp is already unloaded. Withdraw it
        # gently with bounded flexion opening; do not invent contact normals.
        duration=.55 if left_clearance else .30;clear=16.5 if left_clearance else 16.3
        t=np.arange(len(Q))/fps;mask=t>start_s;w=blend((t[mask]-start_s)/duration)
        closed=dict(zip(rig.finger_qpos,d.qpos[fc]));opened=H.open_targets(rig,closed,frac=.90 if left_clearance else .55)
        increment=1.2 if left_clearance else .35
        target=np.array([opened[k] for k in rig.finger_qpos]);target=np.clip(target,d.qpos[fc]-increment,d.qpos[fc]+increment)
        Q[np.ix_(mask,fc)]=(1-w[:,None])*ref['qpos'][idx,fc]+w[:,None]*target
        Q[np.ix_(mask,wc)]=ref['qpos'][idx,wc]
        direction=d.qpos[wc[:3]]-d.xpos[m.body('white_pot').id];direction-=np.dot(direction,up)*up;direction/=np.linalg.norm(direction)
        w=blend((t[mask]-(start_s+duration))/(clear-(start_s+duration)));retreat=(.15*direction+.04*up) if left_clearance else (.10*direction+.10*up)
        Q[np.ix_(mask,wc[:3])]+=w[:,None]*retreat
        lmeta=dict(mode='already unloaded, bounded flexion opening then gravity-frame retreat',measured_left_pot_force_n=leftforce,
            opening_fraction=.90 if left_clearance else .55,maximum_joint_increment_rad=increment,open_duration_s=duration,retreat=retreat.tolist())
    # Correct only the withdrawal direction to the measured gravity frame for
    # the shared helper, whose default retreat assumes world +Z.
    rr=H.HandRig.build(m,'robot_right_','right');rc=np.array([rr.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')]);t=np.arange(len(Q))/fps;mask=t>start_s
    direction=d.qpos[rc[:3]]-d.xpos[m.body('white_pot_lid').id];direction-=np.dot(direction,up)*up;direction/=np.linalg.norm(direction)
    w=blend((t[mask]-(start_s+.30))/(16.3-(start_s+.30)));Q[np.ix_(mask,rc[:3])]=ref['qpos'][idx,rc[:3]]+w[:,None]*(.10*direction+.10*up)
    assert np.array_equal(Q[:idx+1],ref['qpos'][:idx+1]);obj=np.r_[np.arange(56,70)];assert np.array_equal(Q[:,obj],ref['qpos'][:,obj])
    path=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],fps,ref['segment'],ref['segments'],source='Supported final release of both hands, original objects/timing and exact physical prefix',arms_coupled=False,floating_runtime=True)
    execution=execute_reference(scene,path,out/'execution');metric=evaluate_files(spec['source'],execution['trajectory'],spec['manifest'],execution['scene'],spec['geometry'])
    with np.load(execution['trajectory']) as final:
        prefix_exact=bool(np.array_equal(final['substep_qpos'][:count],z['substep_qpos'][:count]));assert prefix_exact
    dump(out/'result.json',dict(reference=str(path),execution=execution,metrics=metric,spec=spec,right_release=meta,left_release=lmeta,prefix_exact=prefix_exact,
        support_path='lid→pot→table',requires_final_release_contact_audit=True))
    print(json.dumps(metric['official_style_reference_proxy'],indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base');p.add_argument('--start-s',type=float,default=15.15);p.add_argument('--left-clearance',action='store_true');run(**vars(p.parse_args()))
