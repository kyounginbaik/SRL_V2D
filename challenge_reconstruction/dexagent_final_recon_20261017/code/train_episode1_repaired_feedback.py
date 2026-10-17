"""Distill a physically cleared episode1 teacher, then execute learned weights."""
from pathlib import Path
import argparse,json
import numpy as np
from repair_episode1_registration import OUT,EP
from learned_feedback_policy import train
from evaluate_policy_component import evaluate
from audit_full_joint_limits import main as audit_joints
from audit_object_support import audit as audit_support
from physics_pickup_search_gravity import dump

def run(teacher=None,gain=.03,centers=1024):
    root=OUT/'patch_height094';teacher=Path(teacher or root/'hands_release_v2');out=root/'learned_feedback'/f'centers_{centers}_gain_{gain:g}'
    support=json.loads((teacher/'release_audit.json').read_text())['summary']
    if support['maximum_hand_object_force_n']>.02 or support['minimum_pot_lid_support_n']<.1 or support['minimum_table_pot_support_n']<.5 or support['maximum_object_linear_speed_m_s']>.02:
        raise ValueError('Teacher must be independently supported and fully clear of both robot hands')
    scene=teacher/'execution/scene_physics.xml';record=teacher/'execution/trajectory.npz'
    train(scene,record,out/'training',centers=centers,feedback_gain=gain,seed=43)
    result=evaluate(1,scene,out/'training/policy.npz',out/'evaluation',EP)
    audit_joints(out/'evaluation');audit_support(out/'evaluation')
    rows=json.loads((out/'evaluation/audit/object_support.json').read_text())['rows'];end=[r for r in rows if r['time_s']>=16.3]
    with np.load(out/'evaluation/execution/trajectory.npz') as z:
        speed=max(float(np.linalg.norm(z['qvel'][i,v:v+3])) for i,t in enumerate(z['timestamps_seconds']) if t>=16.3 for v in (56,62))
    summary=dict(final_window_start_s=16.3,
        max_hand_object_force_n=max(sum(sum(o['fingers'].values()) for o in r['objects'].values()) for r in end),
        min_pot_table_force_n=min(r['objects']['white_pot']['support'].get('table',0.) for r in end),
        min_lid_pot_force_n=min(r['objects']['white_pot_lid']['support'].get('white_pot',0.) for r in end),
        max_object_linear_speed_m_s=speed,reference_proxy_mp_sr=result['metrics']['reference_proxy_mp_sr'],
        teacher=str(teacher),checkpoint=str(out/'training/policy.npz'),provisional_until_visual_and_policy_audit=True)
    dump(out/'evaluation/audit/end_support_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--teacher');p.add_argument('--gain',type=float,default=.03);p.add_argument('--centers',type=int,default=1024);run(**vars(p.parse_args()))
