"""Physics-predicted settling correction for episode1's initial grasp proposals.

Only hand commands change. Initial state, object reference, masses, contact
geometry, actuator force caps, original time axis, and later task targets remain
fixed. A temporary rigid wrist correction fades after attempted engagement.
This is a seed proposal, never a success certificate.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from floating_runtime import execute_reference
from calibrate_controller import controller_scene,tracking_audit
from wrist_reference import continuous_euler_xyz
from evaluate_mp_sr import evaluate_files


def smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


def prepare():
    ep='episode_000001';out=B/'refinement_20261017/settled_seed'/ep;out.mkdir(parents=True,exist_ok=True)
    scene=B/'runs/pilot_v2'/ep/'seed_44/baseline/scene_physics.xml'
    model=mujoco.MjModel.from_xml_path(str(scene))
    original=B/'refinement_20261017/continuous_references'/f'{ep}.npz'
    ref=dict(np.load(original));fps=float(ref['fps']);times=np.arange(len(ref['qpos']))/fps
    intervals=json.loads((B/'refinement_20261017/geometry_v2'/ep/'grasp_intervals.json').read_text())
    chosen=[];seen=set()
    for item in sorted(intervals,key=lambda x:x['start_frame']):
        if item['object'] in seen:continue
        seen.add(item['object']);chosen.append(item)
    # Forecast only the initial settling. Later grasps cannot assume objects
    # remain passive after the robot has already manipulated them.
    horizon=min(len(times),max(x['start_frame'] for x in chosen)+5)
    passive={k:v for k,v in ref.items() if k not in ('qpos','labels','segment')}
    passive['qpos']=ref['qpos'][:horizon].copy()
    robot_cols=[int(model.jnt_qposadr[j]) for j in range(model.njnt) if model.joint(j).name.startswith('robot_')]
    passive['qpos'][:,robot_cols]=ref['qpos'][0,robot_cols]
    passive['labels']=np.array(['passive object settling forecast']*horizon)
    passive['segment']=np.zeros(horizon,dtype=int)
    passive_path=out/'passive_reference.npz';np.savez_compressed(passive_path,**passive)
    forecast_path=out/'passive_forecast/execution_report.json'
    if forecast_path.exists():forecast=json.loads(forecast_path.read_text())
    else:forecast=execute_reference(scene,passive_path,out/'passive_forecast')
    forecast_data=dict(np.load(forecast['trajectory']))
    # Verify the forecast's objects never touch a hand (geometry, every frame).
    d=mujoco.MjData(model);object_bodies={model.body(x['object']).id for x in chosen}
    hand_object_contacts=[]
    for frame,q in enumerate(forecast_data['qpos']):
        d.qpos[:]=q;mujoco.mj_forward(model,d)
        for c in d.contact:
            b1,b2=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
            if (b1 in object_bodies and model.body(b2).name.startswith('robot_')) or (b2 in object_bodies and model.body(b1).name.startswith('robot_')):
                hand_object_contacts.append(frame)
    if hand_object_contacts:raise ValueError('passive settling forecast touched objects with a hand')
    corrections=[]
    for item in chosen:
        frame=item['start_frame'];name=item['object'];j=model.joint(name+'_free').id;adr=int(model.jnt_qposadr[j])
        qref=ref['qpos'][frame,adr:adr+7];actual=forecast_data['qpos'][frame,adr:adr+7]
        rref=Rotation.from_quat(qref[3:7],scalar_first=True)
        ractual=Rotation.from_quat(actual[3:7],scalar_first=True)
        dr=ractual*rref.inv();dt=actual[:3]-dr.apply(qref[:3])
        corrections.append(dict(side=item['side'],object=name,grasp_frame=frame,grasp_seconds=frame/fps,
            original_object_position=qref[:3].tolist(),predicted_object_position=actual[:3].tolist(),
            predicted_translation_error_m=float(np.linalg.norm(actual[:3]-qref[:3])),
            predicted_rotation_error_degrees=float(np.rad2deg(dr.magnitude())),
            correction_rotation_vector=dr.as_rotvec().tolist(),correction_translation=dt.tolist(),
            geometry_seed_applied=bool(item['applied'])))
    seeds={}
    sources={'original':original,'geometry_v2':B/'refinement_20261017/geometry_v2'/ep/'tracking_reference_geometry.npz'}
    for kind,source in sources.items():
        seed=dict(np.load(source));q=seed['qpos'].copy();base=q.copy()
        for correction in corrections:
            side=correction['side'];grasp=correction['grasp_seconds']
            ids=[model.joint(f'robot_{side}_{side}_{k}').id for k in ('pos_x','pos_y','pos_z','rot_x','rot_y','rot_z')]
            cols=model.jnt_qposadr[ids]
            rise=smooth((times-(grasp-.6))/.6)
            fall=1-smooth((times-(grasp+.15))/1.)
            weight=rise*fall
            rotation=Rotation.from_rotvec(weight[:,None]*np.array(correction['correction_rotation_vector']))
            q[:,cols[:3]]=rotation.apply(base[:,cols[:3]])+weight[:,None]*np.array(correction['correction_translation'])
            target_rot=rotation*Rotation.from_euler('XYZ',base[:,cols[3:]])
            q[:,cols[3:]]=continuous_euler_xyz(target_rot,base[0,cols[3:]])
            outside=weight==0;q[np.ix_(outside,cols)]=base[np.ix_(outside,cols)]
        assert np.array_equal(q[0],base[0])
        for name in forecast_data['object_names']:
            adr=int(model.jnt_qposadr[model.joint(str(name)+'_free').id])
            assert np.array_equal(q[:,adr:adr+7],ref['qpos'][:,adr:adr+7])
        # Never alter finger commands in this settling-only seed.
        fingers=[int(model.jnt_qposadr[j]) for j in range(model.njnt)
                 if model.joint(j).name.startswith('robot_') and not model.joint(j).name.endswith(tuple('_'+k for k in ('pos_x','pos_y','pos_z','rot_x','rot_y','rot_z')))]
        assert np.array_equal(q[:,fingers],base[:,fingers])
        for j in range(model.njnt):
            if model.joint(j).name.startswith('robot_'):
                adr=int(model.jnt_qposadr[j]);lo,hi=model.jnt_range[j]
                if np.any(q[:,adr]<lo-1e-8) or np.any(q[:,adr]>hi+1e-8):raise ValueError('seed exceeds robot joint limits')
        seed['qpos']=q;path=out/f'{kind}_settled_reference.npz';np.savez_compressed(path,**seed);seeds[kind]=str(path)
    report=dict(episode=ep,scene=str(scene),passive_forecast=forecast,hand_object_contacts_during_forecast=0,
                corrections=corrections,seeds=seeds,initial_state_unchanged=True,object_reference_unchanged=True,
                original_timing_unchanged=True,finger_commands_unchanged=True,
                correction_ramp_before_grasp_s=.6,hold_after_grasp_s=.15,fade_after_hold_s=1.,
                scope='seed proposal only; no force closure or task success claim')
    (out/'settled_seed_report.json').write_text(json.dumps(report,indent=2));return report


def evaluate(report,kind,variant):
    out=B/'refinement_20261017/settled_seed'/report['episode']/kind/variant;out.mkdir(parents=True,exist_ok=True)
    scene=out/'controller_scene.xml';controller_scene(report['scene'],scene,variant)
    record=execute_reference(scene,report['seeds'][kind],out/'execution')
    metrics=evaluate_files(B.parent/'final_recon'/report['episode'],record['trajectory'],B/'episodes'/report['episode']/'input_manifest.json',record['scene'])
    result=dict(seed=kind,variant=variant,execution=record,metrics=metrics,
                tracking=tracking_audit(record['scene'],record['trajectory'],report['seeds'][kind]))
    (out/'result.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-all',action='store_true');args=p.parse_args()
    report=prepare();print(json.dumps(report,indent=2),flush=True)
    if args.run_all:
        results=[]
        for kind in report['seeds']:
            for variant in ['baseline','kp5000_r200_d5']:
                r=evaluate(report,kind,variant);results.append(r)
                print(kind,variant,r['metrics']['reference_proxy_mp_sr'],flush=True)
        (B/'refinement_20261017/settled_seed'/report['episode']/'summary.json').write_text(json.dumps(results,indent=2))
