"""Execute collision-surface lid grasp poses with bounded robot-only feedback."""
from pathlib import Path
import argparse,json,sys
import numpy as np,mujoco
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H
from lid_collision_grasps import collision_surface,SCENE
from contact_retention_runtime import execute_retention
from evaluate_mp_sr import evaluate_files
from audit_acquisition import audit


def run(task):
    candidate=task//2;opening=(.006,.010)[task%2]
    inputs=B/'refinement_20261017/lid_collision_grasps/episode_000001'
    model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model)
    data.qpos[:]=np.load(inputs/f'pose_{candidate:02d}.npz')['qpos'];mujoco.mj_forward(model,data)
    rig=H.HandRig.build(model,'robot_right_','right');lid=model.body('white_pot_lid').id
    op=data.xpos[lid].copy();R=data.xmat[lid].reshape(3,3).copy()
    wcols=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')])
    localp=R.T@(data.qpos[wcols[:3]]-op)
    localR=R.T@Rotation.from_euler('XYZ',data.qpos[wcols[3:]]).as_matrix()
    closed={k:float(data.qpos[a]) for k,a in rig.finger_qpos.items()}
    points,normals,_=collision_surface(model,data,lid)
    p=data.site_xpos[rig.fingertip_sites['thumb']].copy()
    ni=cKDTree(points).query((p-op)@R)[1];normal=R@normals[ni]
    residual=H.finger_ik(model,data,rig,{'thumb':p+opening*normal},iters=100)
    opened={k:float(data.qpos[a]) for k,a in rig.finger_qpos.items() if H.finger_of_joint(k)=='thumb'}
    reference=B/'refinement_20261017/lid_normal/episode_000001/candidate_02_depth3/reference.npz'
    config=dict(side='right',object_name='white_pot_lid',
        object_from_wrist_position=localp.tolist(),object_from_wrist_rotation=localR.tolist(),
        closed_finger_targets=closed,open_thumb_targets=opened,
        acquisition_start_s=1.1,deadline_s=2.3,approach_s=.35,
        opposing_fingers=['index'],min_contact_force_n=.05,contact_confirm_s=.02,
        capture_confirm_s=.06,thumb_closure_s=.15,dwell_s=.12,
        transport_blend_s=.6,max_tracking_offset_m=.10,
        release_start_s=4.45,release_blend_s=.25,require_opposed_contact_geometry=True,
        requested_thumb_opening_m=opening,thumb_opening_ik_residual_m=residual,
        actual_thumb_outward_travel_m=float((data.site_xpos[rig.fingertip_sites['thumb']]-p)@normal),
        collision_planned_pose=str(inputs/f'pose_{candidate:02d}.npz'),source_reference=str(reference))
    out=B/f'refinement_20261017/lid_collision_feedback/episode_000001/candidate_{candidate:02d}_open{round(opening*1000)}'
    out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(config,indent=2))
    execution=execute_retention(SCENE,reference,out/'execution',config)
    metrics=evaluate_files(B.parent/'final_recon/episode_000001',execution['trajectory'],
        B/'episodes/episode_000001/input_manifest.json',execution['scene'])
    retention=audit(out/'execution')
    report=dict(candidate=candidate,config=config,execution=execution,metrics=metrics,retention=retention)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(metrics['official_style_reference_proxy'],indent=2))
    print(json.dumps({k:v for k,v in retention.items() if k!='table_contact_and_height_samples'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=int,required=True)
    run(parser.parse_args().task)
