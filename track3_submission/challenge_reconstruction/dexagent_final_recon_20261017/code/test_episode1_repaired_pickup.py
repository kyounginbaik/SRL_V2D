"""Physical lid-grasp replay on an explicitly versioned reconstruction candidate.

One fixed initial scene uses deterministic passive seating, with image evidence
reported. Original objects, source trajectories, and previous trials are intact.
No object is actuated or updated from reference poses during execution.
"""
import argparse,copy,json,xml.etree.ElementTree as ET
import cv2,mujoco,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,EP,OUT,load,outline
from floating_runtime import execute_reference
from validate_episode1_repair import values


def run(horizon=4.,align_lid=False):
    parent=OUT/'patch_height094';folder=parent/(f'pickup_{horizon:g}s'+('_aligned_lid_v2' if align_lid else ''));folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists():raise FileExistsError(folder)
    source_scene=parent/'passive/scene_physics.xml';source=mujoco.MjModel.from_xml_path(str(source_scene))
    seated=np.load(parent/'passive/trajectory.npz')['qpos'][-1];q=source.key_qpos[0].copy()
    root=ET.parse(source_scene).getroot();poses={}
    for name in ['white_pot','white_pot_lid']:
        a=int(source.joint(name+'_free').qposadr[0]);q[a:a+7]=seated[a:a+7];pose=q[a:a+7];poses[name]=pose
        body=root.find(f".//body[@name='{name}']");body.set('pos',values(pose[:3]));body.set('quat',values(pose[3:]))
    root.find('keyframe/key').set('qpos',values(q))
    # Reuse the exact already-audited controller of the genuine opposed grasp.
    donor=B/'refinement_20261018/episode_000001/pickup_cem_v1/pose_01/execution/scene_physics.xml'
    old=root.find('actuator');root.remove(old);root.append(copy.deepcopy(ET.parse(donor).getroot().find('actuator')))
    scene=folder/'scene.xml';ET.indent(root);ET.ElementTree(root).write(scene,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(scene));donor_model=mujoco.MjModel.from_xml_path(str(donor))
    assert np.array_equal(model.actuator_forcerange,donor_model.actuator_forcerange)
    assert np.array_equal(model.actuator_gainprm,donor_model.actuator_gainprm)
    assert np.array_equal(model.actuator_biasprm,donor_model.actuator_biasprm)
    # Immutable source-image audit before simulation.
    _,_,cam,_=load();rgb=cv2.imread(str(OUT/'source_000.jpg'));shift=np.array(json.loads((EP/'input_manifest.json').read_text())['setup_from_source'])[:3,3]
    mesh_paths={'white_pot':OUT/'pot_geometry_rgb_depth_height094/white_pot_visual.obj','white_pot_lid':B.parent/'final_recon/episode_000001/assets/white_pot_lid_visual.obj'}
    reports={};panel=rgb.copy()
    for name,path in mesh_paths.items():
        mesh=trimesh.load(path,force='mesh',process=False);sample,_=trimesh.sample.sample_surface(mesh,220000,seed=22);pose=poses[name];R=Rotation.from_quat(pose[3:],scalar_first=True).as_matrix()
        pc=np.load(OUT/f'{name}_depth_points.npz')['points_source']+shift;dist=cKDTree(sample).query((pc-pose[:3])@R)[0]
        reports[name]=dict(pose=pose.tolist(),median_stereo_surface_m=float(np.median(dist)),p90_stereo_surface_m=float(np.quantile(dist,.9)))
        outline(panel,mesh,pose,cam,(0,255,0) if name=='white_pot' else (255,160,0))
    cv2.putText(panel,'Versioned physical seat: pot green; lid blue',(620,725),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
    cv2.imwrite(str(folder/'settled_rgb_overlay.jpg'),panel[680:1510,600:1500])
    (folder/'initial_audit.json').write_text(json.dumps(dict(source_scene=str(source_scene),seating='Same deterministic passive seat for every candidate, selected for source alignment audit; original source unchanged',
        images=reports,original_object_proxy_unchanged=True,controller_source=str(donor),provisional=True),indent=2))
    reference_path=B/'refinement_20261018/episode_000001/pickup_cem_v1/pose_01/reference.npz'
    with np.load(reference_path) as z:reference={k:z[k] for k in z.files}
    frames=min(len(reference['qpos']),int(round(horizon*float(reference['fps'])))+1)
    reference['qpos']=reference['qpos'][:frames]
    if 'labels' in reference:reference['labels']=reference['labels'][:frames]
    if align_lid:
        from wrist_reference import continuous_euler_xyz
        previous=np.load(B/'refinement_20261018/episode_000001/pickup_cem_v1/pose_01/execution/trajectory.npz')
        lid_address=int(model.joint('white_pot_lid_free').qposadr[0]);old_lid=previous['qpos'][28,lid_address:lid_address+7]
        new_lid=poses['white_pot_lid'];delta_R=Rotation.from_quat(new_lid[3:],scalar_first=True)*Rotation.from_quat(old_lid[3:],scalar_first=True).inv()
        delta_p=new_lid[:3]-delta_R.apply(old_lid[:3]);t=np.arange(frames)/float(reference['fps'])
        smooth=lambda z: np.clip(z,0.,1.)**3*(10.-15.*np.clip(z,0.,1.)+6.*np.clip(z,0.,1.)**2)
        weight=smooth((t-1.)/.7)*(1.-smooth((t-2.2)/1.))
        wrist=[int(model.joint('robot_right_right_'+s).qposadr[0]) for s in ['pos_x','pos_y','pos_z','rot_x','rot_y','rot_z']]
        Q=reference['qpos'].copy();old=Q[:,wrist].copy();desired=delta_R.apply(old[:,:3])+delta_p
        Q[:,wrist[:3]]=old[:,:3]+weight[:,None]*(desired-old[:,:3])
        new_rot=Rotation.from_rotvec(weight[:,None]*delta_R.as_rotvec())*Rotation.from_euler('XYZ',old[:,3:])
        Q[:,wrist[3:]]=continuous_euler_xyz(new_rot,initial_angles=old[0,3:]);reference['qpos']=Q
        (folder/'wrist_rebase.json').write_text(json.dumps(dict(old_lid_pose=old_lid.tolist(),new_lid_pose=new_lid.tolist(),
            full_correction_translation_m=delta_p.tolist(),full_correction_rotation_deg=float(np.rad2deg(delta_R.magnitude())),
            envelope='C2 rise1.0–1.7s, hold until2.2s, C2 fade by3.2s; original object targets and clock immutable'),indent=2))
    execution=execute_reference(scene,reference,folder/'execution',expected_duration_s=(frames-1)/float(reference['fps']))
    actual=np.load(execution['trajectory']);names=list(actual['object_names'].astype(str));out=[]
    # Prefix position error is diagnostic only; full-horizon MP-SR remains the
    # final criterion and is never inferred from this pickup test.
    original=np.load(B.parent/'final_recon/episode_000001/object_poses.npz');original_names=list(original['names'].astype(str));target=original['world_from_object'][:frames]
    for name in names:
        k=names.index(name);j=original_names.index(name);error=np.linalg.norm(actual['object_positions'][:,k]-target[:,j,:3,3]-shift,axis=1)
        out.append(dict(object=name,prefix_mean_position_error_m=float(error.mean()),last_position_error_m=float(error[-1])))
    report=dict(initial=reports,execution=execution,prefix_seconds=(frames-1)/float(reference['fps']),objects=out,
        task_success_not_inferred=True,original_reference=str(reference_path))
    (folder/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--horizon',type=float,default=4.);p.add_argument('--align-lid',action='store_true');run(**vars(p.parse_args()))
