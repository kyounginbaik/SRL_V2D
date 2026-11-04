"""Stage correct repaired visuals with explicit source/reconstruction lineage.

This does not modify targets, scenes, checkpoints, or execute a submission.
The mesh exported by the packer must equal the mesh rendered by the simulator.
"""
from pathlib import Path
import json,shutil,xml.etree.ElementTree as ET
import numpy as np,mujoco,trimesh
from scipy.spatial import cKDTree
from learned_feedback_policy import sha
B=Path(__file__).resolve().parents[1]
ROOT=B/'refinement_20261018/episode_000001/reconstruction_repair/patch_height094'


def main():
    source=B/'episodes/episode_000001';out=ROOT/'repaired_adapter'
    scene=ROOT/'hands_release_v2/execution/scene_physics.xml'
    if (out/'input_manifest.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    old=json.loads((source/'input_manifest.json').read_text());manifest=json.loads(json.dumps(old))
    for f in ['tracking_reference.npz','metric_geometry.json','camera_trajectory.npz']:
        shutil.copy2(source/f,out/f)
        assert sha(source/f)==sha(out/f)
    (out/'setup').mkdir(exist_ok=True);shutil.copy2(scene,out/'setup/scene.xml')
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,d,0);mujoco.mj_forward(m,d)
    xml=ET.parse(scene).getroot();meshes={x.get('name'):x for x in xml.findall('./asset/mesh')};lineage={}
    for obj in manifest['objects']:
        name=obj['id'];body=xml.find(f'.//body[@name="{name}"]');items=[];visuals=[]
        for geom in body.findall('geom'):
            if not geom.get('mesh'):continue
            mesh=meshes[geom.get('mesh')];path=Path(mesh.get('file')).resolve()
            kind='visual' if geom.get('group')=='1' else 'collision'
            items.append(dict(source=str(path),sha256=sha(path),kind=kind,scene_geom=geom.get('name')))
            if kind=='visual':visuals.append((geom,mesh,path))
        if len(visuals)!=1:raise ValueError('Expected one visual mesh per object')
        geom,mesh,path=visuals[0]
        if mesh.get('scale','1 1 1')!='1 1 1' or geom.get('pos','0 0 0')!='0 0 0' or geom.get('quat','1 0 0 0')!='1 0 0 0':
            raise ValueError('Export requires baked mesh scale and identity visual-local transform')
        # MuJoCo internally recenters mesh vertices. Undo that via compiled
        # geometry/body transforms and compare to the actual body-frame file.
        gid=m.geom(geom.get('name')).id;mid=int(m.geom_dataid[gid]);bid=m.body(name).id
        v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].astype(float)
        world=v@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
        body_v=(world-d.xpos[bid])@d.xmat[bid].reshape(3,3)
        raw=trimesh.load(path,force='mesh',process=False).vertices
        error=max(cKDTree(raw).query(body_v)[0].max(),cKDTree(body_v).query(raw)[0].max())
        if error>1e-6:raise ValueError(f'Simulator/export body-frame mismatch: {name}: {error}')
        prior=[x for x in obj['meshes'] if x['kind']=='visual'][0]
        lineage[name]=dict(original_visual=prior,repaired_visual=[x for x in items if x['kind']=='visual'][0],
            compiled_body_frame_vertex_hausdorff_m=float(error),scale_baked_into_vertices=True,
            changed_visual=sha(path)!=prior['sha256'],collision_part_count=sum(x['kind']=='collision' for x in items),
            original_source_urdf=obj['source_urdf'],original_source_urdf_sha256=obj['source_urdf_sha256'])
        obj['source_meshes_before_repair']=obj['meshes'];obj['meshes']=items
        obj['source_asset_xml_before_repair']=obj.pop('asset_xml',None)
        obj['simulation_scene']=str((out/'setup/scene.xml').resolve())
        obj['simulation_body']=name;obj['mesh_lineage']=lineage[name]
    manifest['original_adapter_manifest']=str((source/'input_manifest.json').resolve())
    manifest['original_adapter_manifest_sha256']=sha(source/'input_manifest.json')
    manifest['original_setup_before_repair']=manifest['setup'];manifest['setup']=str((out/'setup').resolve())
    manifest['source_scene_recon_before_repair']=manifest.pop('scene_recon')
    manifest['simulation_scene']=str((out/'setup/scene.xml').resolve());manifest['simulation_scene_sha256']=sha(scene)
    manifest['support_before_repair']=manifest['support']
    manifest['support']=dict(method='Independent calibrated stereo tabletop plane and deterministic passive seated initial objects',
        evidence=str((B/'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json').resolve()),
        gravity_m_s2=m.opt.gravity.tolist(),table_body_position=d.xpos[m.body('table').id].tolist(),
        table_body_rotation=d.xmat[m.body('table').id].reshape(3,3).tolist())
    for side in ['left','right']:
        oldservo=manifest['robot'][side]['wrist_servo'];manifest['robot'][side]['wrist_servo_before_repair']=oldservo
        aids=[]
        for name in ['pos_x','pos_y','pos_z','rot_x','rot_y','rot_z']:
            j=m.joint(f'robot_{side}_{side}_{name}').id;aids.append(int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]))
        manifest['robot'][side]['wrist_servo']=dict(joint_names=[m.joint(int(m.actuator_trnid[a,0])).name for a in aids],
            kp=m.actuator_gainprm[aids,0].tolist(),kv=(-m.actuator_biasprm[aids,2]).tolist(),force_ranges=m.actuator_forcerange[aids].tolist(),
            limitation='Effort bounded; MuJoCo position servos have no hard velocity bound')
    manifest['camera']['trajectory_npz']=str((out/'camera_trajectory.npz').resolve())
    manifest['reconstruction_repair']=dict(version='episode1-rgbd-shell-height094-fixedseat',
        scene_sha256=sha(scene),lineage=lineage,original_reference_poses_and_timestamps_unchanged=True,
        initial_objects='One deterministic source-validated passive seat; distinct from known-invalid original reconstructed initial object poses',
        targets='Original tracking_reference/object_poses remain scoring targets, not physical-state initialization',
        source_evidence=str((ROOT.parent/'status.json').resolve()),
        limitations=['Source inertial tensors retained despite small pot shape repair.',
                    'Lid visual topology remains malformed; patch collisions cover source surface triangles without inferred watertight fill.',
                    'Original initial-aligned proxy currently fails; this adapter is not submission approval.'])
    (out/'mesh_export_audit.json').write_text(json.dumps(dict(scene=str(scene),scene_sha256=sha(scene),objects=lineage,
        original_targets_unchanged=True,ready_for_final_submission=False),indent=2)+'\n')
    (out/'input_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(adapter=str(out),objects=lineage),indent=2))
if __name__=='__main__':main()
