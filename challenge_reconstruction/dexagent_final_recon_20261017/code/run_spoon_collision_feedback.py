"""Physically acquire collision-qualified neck/bowl grasps; robot commands only."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from contact_retention_smooth_runtime import execute_retention
from evaluate_mp_sr import evaluate_files
from tool_library.common import hand as H

B=Path(__file__).resolve().parents[1];R=B/'refinement_20261017'

def run(rank):
    ep='episode_000003';source=R/'spoon_collision_grasps'/ep
    scene=R/'cem_v2'/ep/'execution/scene_physics.xml'
    reference=R/'cem_v2'/ep/'reference.npz'
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    rig=H.HandRig.build(model,'robot_right_','right')
    pose=source/f'pose_{rank:02d}.npz';data.qpos[:]=np.load(pose)['qpos'];mujoco.mj_forward(model,data)
    report=json.loads((source/'report.json').read_text())
    geometry=next(row['geometry'] for row in report['exported'] if row['rank']==rank)
    patches=geometry['actual_contacts']['patches'];best=None
    for a in patches['thumb']:
        for b in patches['index']:
            na,nb=np.asarray(a['normal']),np.asarray(b['normal'])
            delta=np.asarray(a['position'])-np.asarray(b['position']);line=delta/np.linalg.norm(delta)
            margin=min(-.5-na@nb,na@line-1/np.sqrt(1+1.2**2),nb@(-line)-1/np.sqrt(1+1.2**2))
            if best is None or margin>best[0]:best=(float(margin),na,a,b)
    if best is None or best[0]<0:raise ValueError('pose does not pass actual collision-contact friction cones')
    wc=np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']])
    oc=int(model.joint('wooden_spoon_free').qposadr[0]);objectR=Rotation.from_quat(data.qpos[oc+3:oc+7],scalar_first=True)
    localp=objectR.inv().apply(data.qpos[wc[:3]]-data.qpos[oc:oc+3]);localR=objectR.inv()*Rotation.from_euler('XYZ',data.qpos[wc[3:]])
    closed={key:float(data.qpos[col]) for key,col in rig.finger_qpos.items()}
    sid=rig.fingertip_sites['thumb'];tip=data.site_xpos[sid].copy();normal=best[1]
    H.finger_ik(model,data,rig,{'thumb':tip+.008*normal},iters=100)
    opened={key:float(data.qpos[col]) for key,col in rig.finger_qpos.items() if H.finger_of_joint(key)=='thumb'}
    actual_open=float((data.site_xpos[sid]-tip)@normal)
    config=json.loads((R/'contact_acquisition'/ep/'config_1opp.json').read_text())
    config.update(object_from_wrist_position=localp.tolist(),object_from_wrist_rotation=localR.as_matrix().tolist(),
        closed_finger_targets=closed,open_thumb_targets=opened,opposing_fingers=['index'],
        acquisition_start_s=1.3,approach_s=.45,release_start_s=14.45,release_blend_s=.35,
        transport_blend_s=.8,require_opposed_contact_geometry=True,capture_preload_m=.002,capture_preload_rotation_deg=3.,
        geometry_source={'pose':str(pose),'sha256':hashlib.sha256(pose.read_bytes()).hexdigest(),
                         'actual_contact_cone_margin':best[0],'thumb_open_requested_m':.008,'thumb_open_achieved_m':actual_open})
    out=R/'spoon_collision_feedback'/ep/f'pose_{rank:02d}';out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    execution=execute_retention(scene,reference,out/'execution',config)
    inp=B/'episodes'/ep;manifest=json.loads((inp/'input_manifest.json').read_text())
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    report={'execution':execution,'metrics':metrics,'reference':str(reference),'config':config,
            'left_commands_preserved':'all 28 original CEM left commands','finger_retention':'collision-qualified closed target, not measured-finger reset'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report':str(out/'report.json'),'physics_verified':execution['physics_verified'],
        'events':execution['feedback_events'],'proxy_mp_sr':metrics['reference_proxy_mp_sr']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rank',type=int,choices=[0,1],required=True);a=p.parse_args();run(a.rank)
