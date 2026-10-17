"""Configure the shared robot-only SE(3) contact acquisition controller for lid."""
from pathlib import Path
import argparse,json,sys
import numpy as np,mujoco,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from tool_library.common import hand as H
from evaluate_mp_sr import evaluate_files


def make_config(candidate,opening):
    inp=B/f'refinement_20261017/lid_normal/episode_000001/candidate_{candidate:02d}_depth3'
    ref=dict(np.load(inp/'reference.npz'));scene=inp/'execution/scene_physics.xml'
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    rig=H.HandRig.build(model,'robot_right_','right');frame=round(1.7*float(ref['fps']))
    data.qpos[:]=ref['qpos'][frame]
    forecast=np.load(B/'refinement_20261017/settled_seed/episode_000001/passive_forecast/trajectory.npz')
    for name in forecast['object_names']:
        adr=int(model.jnt_qposadr[model.joint(str(name)+'_free').id])
        data.qpos[adr:adr+7]=forecast['qpos'][frame,adr:adr+7]
    mujoco.mj_forward(model,data);lid=model.body('white_pot_lid').id
    op=data.xpos[lid].copy();rot=data.xmat[lid].reshape(3,3).copy()
    wcols=np.array([rig.wrist_qpos[k] for k in ('x','y','z','rx','ry','rz')])
    localp=rot.T@(data.qpos[wcols[:3]]-op)
    localr=rot.T@Rotation.from_euler('XYZ',data.qpos[wcols[3:]]).as_matrix()
    closed={k:float(data.qpos[a]) for k,a in rig.finger_qpos.items()}
    mesh=trimesh.load(B.parent/'final_recon/episode_000001/assets/white_pot_lid_visual.obj',force='mesh',process=False)
    mesh.fix_normals(multibody=True);points,faces=trimesh.sample.sample_surface(mesh,60000,seed=43)
    normals=mesh.face_normals[faces];mask=points[:,0]>.018;points=points[mask];normals=normals[mask]
    p=data.site_xpos[rig.fingertip_sites['thumb']].copy();idx=cKDTree(points).query((p-op)@rot)[1]
    normal=rot@normals[idx]
    residual=H.finger_ik(model,data,rig,{'thumb':p+opening*normal},iters=100)
    opened={k:float(data.qpos[a]) for k,a in rig.finger_qpos.items() if H.finger_of_joint(k)=='thumb'}
    config=dict(side='right',object_name='white_pot_lid',
        object_from_wrist_position=localp.tolist(),object_from_wrist_rotation=localr.tolist(),
        closed_finger_targets=closed,open_thumb_targets=opened,
        acquisition_start_s=1.1,deadline_s=2.3,approach_s=.35,
        opposing_fingers=['index'],min_contact_force_n=.05,contact_confirm_s=.02,
        capture_confirm_s=.06,thumb_closure_s=.15,dwell_s=.12,
        transport_blend_s=.6,max_tracking_offset_m=.10,
        requested_thumb_opening_m=opening,thumb_opening_ik_residual_m=residual,
        actual_thumb_outward_travel_m=float((data.site_xpos[rig.fingertip_sites['thumb']]-p)@normal),
        source_reference=str(inp/'reference.npz'))
    return scene,inp/'reference.npz',config


def run(task,tag='lid_acquisition_v2'):
    from contact_acquisition_runtime import execute_acquisition
    candidate=task//2;opening=(.006,.010)[task%2]
    scene,reference,config=make_config(candidate,opening)
    out=B/f'refinement_20261017/{tag}/episode_000001/candidate_{candidate:02d}_open{round(opening*1000)}'
    out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(config,indent=2))
    execution=execute_acquisition(scene,reference,out/'execution',config)
    metrics=evaluate_files(B.parent/'final_recon/episode_000001',execution['trajectory'],
        B/'episodes/episode_000001/input_manifest.json',execution['scene'])
    report=dict(candidate=candidate,config=config,execution=execution,metrics=metrics)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(metrics['official_style_reference_proxy'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',type=int,required=True)
    parser.add_argument('--tag',default='lid_acquisition_v2')
    args=parser.parse_args();run(args.task,args.tag)
