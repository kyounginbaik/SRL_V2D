"""Seat the physically held lid by robot motion before a supported release."""
import argparse,json
import numpy as np
import mujoco
from repair_episode1_registration import OUT
from physics_pickup_search_gravity import H,TR,dump,blend,execute_reference,evaluate_files

def run(depth_mm=10.,start_s=14.1,complete_s=14.9):
    root=OUT/'patch_height094';base=root/'regrasp_actual_cem_v1';out=root/f'lid_seat_{depth_mm:04.1f}mm';out.mkdir(exist_ok=True)
    r=json.loads((base/'result.json').read_text());spec=r['spec'];m=mujoco.MjModel.from_xml_path(r['execution']['scene']);ref=TR.load_reference(base/'reference.npz',m);Q=ref['qpos'].copy()
    rig=H.HandRig.build(m,'robot_right_','right');wc=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')]);fc=np.array(list(rig.finger_qpos.values()));t=np.arange(len(Q))/ref['fps'];i=round(start_s*ref['fps']);up=-m.opt.gravity/np.linalg.norm(m.opt.gravity)
    mask=t>start_s;w=blend((t[mask]-start_s)/(complete_s-start_s))
    Q[np.ix_(mask,wc)]=Q[i,wc];Q[np.ix_(mask,fc)]=Q[i,fc];Q[np.ix_(mask,wc[:3])]-=w[:,None]*depth_mm/1000*up
    assert np.array_equal(Q[:i+1],ref['qpos'][:i+1]);other=np.setdiff1d(np.arange(m.nq),np.r_[wc,fc]);assert np.array_equal(Q[:,other],ref['qpos'][:,other])
    path=TR.save_reference(out/'reference.npz',m,Q,ref['labels'],ref['fps'],ref['segment'],ref['segments'],source='Robot-only slow lid seating before release; no object state or target changes',arms_coupled=False,floating_runtime=True)
    execution=execute_reference(spec['scene'],path,out/'execution');metric=evaluate_files(spec['source'],execution['trajectory'],spec['manifest'],execution['scene'],spec['geometry'])
    result=dict(reference=str(path),execution=execution,metrics=metric,spec=spec,placement=dict(depth_mm=depth_mm,start_s=start_s,complete_s=complete_s,up=up.tolist()),task_success=False,scope='support acquisition diagnostic; hands still hold until release stage')
    dump(out/'result.json',result);print(json.dumps(metric['official_style_reference_proxy'],indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--depth-mm',type=float,default=10.);p.add_argument('--start-s',type=float,default=14.1);p.add_argument('--complete-s',type=float,default=14.9);run(**vars(p.parse_args()))
