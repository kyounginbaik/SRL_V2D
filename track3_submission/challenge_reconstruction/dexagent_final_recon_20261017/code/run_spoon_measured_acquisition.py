"""Physical acquisition tests using measured grasps; robot commands only."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from contact_retention_measured_runtime import execute_retention
from evaluate_mp_sr import evaluate_files
from tool_library.common import hand as H

B=Path(__file__).resolve().parents[1]
R=B/'refinement_20261017'

def donor_config():
    root=R/'block_polish/episode_000005'
    model=mujoco.MjModel.from_xml_path(str(root/'execution/scene_physics.xml'))
    data=mujoco.MjData(model);rig=H.HandRig.build(model,'robot_right_','right')
    rec=np.load(root/'execution/trajectory.npz')
    k=int(np.argmin(abs(rec['substep_time_s']-4.4)))
    data.qpos[:]=rec['substep_qpos'][k];data.qvel[:]=rec['substep_qvel'][k];data.ctrl[:]=rec['substep_ctrl'][k]
    mujoco.mj_forward(model,data)
    wc=np.array([rig.wrist_qpos[k] for k in ['x','y','z','rx','ry','rz']])
    oc=int(model.joint('wooden_spoon_free').qposadr[0]);ob=model.body('wooden_spoon').id
    objectR=Rotation.from_quat(data.qpos[oc+3:oc+7],scalar_first=True)
    wristR=Rotation.from_euler('XYZ',data.qpos[wc[3:]])
    localp=objectR.inv().apply(data.qpos[wc[:3]]-data.qpos[oc:oc+3]);localR=objectR.inv()*wristR
    closed={key:float(data.qpos[col]) for key,col in rig.finger_qpos.items()}
    thumb_normals=[]
    for contact in data.contact:
        b1,b2=int(model.geom_bodyid[contact.geom1]),int(model.geom_bodyid[contact.geom2])
        if b1==ob:hand,sign=b2,1.
        elif b2==ob:hand,sign=b1,-1.
        else:continue
        if '_thumb_' in model.body(hand).name:thumb_normals.append(sign*np.array(contact.frame[:3]))
    if not thumb_normals:raise ValueError('donor has no thumb contact')
    normal=np.mean(thumb_normals,axis=0);normal/=np.linalg.norm(normal)
    sid=rig.fingertip_sites['thumb'];old=data.site_xpos[sid].copy()
    H.finger_ik(model,data,rig,{'thumb':old+.010*normal},iters=100)
    opened={key:float(data.qpos[col]) for key,col in rig.finger_qpos.items() if H.finger_of_joint(key)=='thumb'}
    provenance={'source_episode':'episode_000005','source_time_s':float(rec['substep_time_s'][k]),
        'source_trajectory_sha256':hashlib.sha256((root/'execution/trajectory.npz').read_bytes()).hexdigest(),
        'source_pose':'actual physical qpos, not reference command','requested_thumb_outward_m':.010,
        'achieved_thumb_outward_m':float((data.site_xpos[sid]-old)@normal)}
    config=json.loads((R/'contact_acquisition/episode_000003/config_1opp.json').read_text())
    config.update(object_from_wrist_position=localp.tolist(),object_from_wrist_rotation=localR.as_matrix().tolist(),
        closed_finger_targets=closed,open_thumb_targets=opened,acquisition_start_s=1.3,approach_s=.45,
        donor_provenance=provenance)
    return config

def run(kind):
    if kind=='geometry':
        config=json.loads((R/'contact_acquisition/episode_000003/config_1opp.json').read_text())
        reference=R/'geometry_combo_v8_clamp/episode_000003/tracking_reference_geometry.npz'
    else:
        config=donor_config();reference=R/'cem_v2/episode_000003/reference.npz'
    config.update(release_start_s=14.45,release_blend_s=.35,require_opposed_contact_geometry=True,
                  capture_preload_m=.002,capture_preload_rotation_deg=3.)
    ep='episode_000003';scene=R/'cem_v2'/ep/'execution/scene_physics.xml'
    out=R/'contact_retention_measured'/ep/kind;out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    execution=execute_retention(scene,reference,out/'execution',config)
    inp=B/'episodes'/ep;manifest=json.loads((inp/'input_manifest.json').read_text())
    metrics=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    report={'execution':execution,'metrics':metrics,'reference':str(reference),'config':config}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report':str(out/'report.json'),'physics_verified':execution['physics_verified'],
                      'events':execution['feedback_events'],'proxy_mp_sr':metrics['reference_proxy_mp_sr']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['geometry','donor'],required=True);a=p.parse_args();run(a.kind)
