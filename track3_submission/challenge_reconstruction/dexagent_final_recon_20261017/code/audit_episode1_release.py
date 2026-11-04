"""Read exact physical contact support before planning final hand release."""
import argparse,json
from pathlib import Path
import mujoco,numpy as np
from repair_episode1_registration import OUT
from physics_pickup_search_gravity import dump

def run(base=None):
    folder=Path(base or OUT/'patch_height094/regrasp_actual_cem_v1');r=json.loads((folder/'result.json').read_text());m=mujoco.MjModel.from_xml_path(r['execution']['scene']);z=np.load(r['execution']['trajectory'])
    d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['initial_state'],int(z['state_spec']));rows=[];force=np.zeros(6)
    for k,ctrl in enumerate(z['substep_ctrl']):
        d.ctrl[:]=ctrl;mujoco.mj_step(m,d)
        if d.time<14.4 or (k+1)%25:continue
        row=dict(time_s=float(d.time),contacts={},objects={})
        for i,c in enumerate(d.contact):
            a,b=m.body(m.geom_bodyid[c.geom1]).name,m.body(m.geom_bodyid[c.geom2]).name
            if not any(x in ('white_pot','white_pot_lid') for x in (a,b)):continue
            mujoco.mj_contactForce(m,d,i,force);f=max(0.,float(force[0]));key=' / '.join(sorted((a,b)))
            if f>.02:row['contacts'][key]=row['contacts'].get(key,0.)+f
        for name in ('white_pot','white_pot_lid'):
            j=m.joint(name+'_free');v=d.qvel[int(j.dofadr[0]):int(j.dofadr[0])+6]
            row['objects'][name]=dict(position=d.xpos[m.body(name).id].tolist(),linear_speed=float(np.linalg.norm(v[:3])),angular_speed=float(np.linalg.norm(v[3:])))
        rows.append(row)
    final=[row for row in rows if row['time_s']>=16.3];handforces=[sum(v for k,v in row['contacts'].items() if 'robot_' in k) for row in final]
    summary=dict(final_window_start_s=16.3,maximum_hand_object_force_n=max(handforces,default=None),
        maximum_object_linear_speed_m_s=max((o['linear_speed'] for row in final for o in row['objects'].values()),default=None),
        minimum_table_pot_support_n=min((row['contacts'].get('table / white_pot',0.) for row in final),default=None),
        minimum_pot_lid_support_n=min((row['contacts'].get('white_pot / white_pot_lid',0.) for row in final),default=None))
    out=folder/'release_audit.json';dump(out,dict(rows=rows,summary=summary,exact_replay_qpos_max_error=float(np.max(abs(d.qpos-z['final_qpos'])))))
    for row in rows:
        if round(row['time_s']*20)%2==0:print(json.dumps(row),flush=True)

    print('SUMMARY',json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base');run(**vars(p.parse_args()))
