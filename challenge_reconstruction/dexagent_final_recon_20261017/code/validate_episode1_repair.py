"""Build and independently replay a versioned passive repaired-scene test."""
from pathlib import Path
import argparse,copy,hashlib,json,sys,xml.etree.ElementTree as ET
import numpy as np
import mujoco,trimesh
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,EP,OUT,SCENE,align_axis
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))
from floating_runtime import execute_reference


def values(x):return ' '.join(f'{float(v):.15g}' for v in np.asarray(x).ravel())


def prepare(mode,version='pot_geometry_rgb_depth'):
    suffix=version.removeprefix('pot_geometry_rgb_depth')
    out=OUT/(mode+suffix);out.mkdir(parents=True,exist_ok=True)
    fit=json.loads((OUT/version/'fit.json').read_text())
    support=json.loads((B/'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json').read_text())
    n=np.array(support['fitted_plane_normal_source']);T=np.array(support['proposed_setup_from_source'])
    manifest=json.loads((EP/'input_manifest.json').read_text());shift=np.array(manifest['setup_from_source'])[:3,3]
    offset=T[2,3]-n@shift
    root=ET.parse(SCENE).getroot();root.find('option').set('gravity',values(-9.81*n))
    table=root.find(".//geom[@name='g_table']");old=np.fromstring(table.get('pos'),sep=' ');half=float(table.get('size').split()[1])
    center=np.r_[old[:2],(-offset-n[:2]@old[:2])/n[2]]
    table.set('pos',values(center-half*n));table.set('quat',values(align_axis(np.array([0.,0.,1.]),n).as_quat(scalar_first=True)))
    poses={'white_pot':np.array(fit['pose']),'white_pot_lid':np.array(fit['unchanged_lid_pose'])}
    axis=np.array(fit['local_height_axis']);floor=fit['local_floor'];x=np.array(fit['parameters'])
    def warp(v):
        h=v@axis-floor;w=np.clip((v[:,2]-.075)/.055,0,1);w=w*w*(3-2*w)
        result=v+((x[3]-1)*h*(1-w)+w*(x[4]+x[5]*(v[:,2]-.13)))[:,None]*axis
        if len(x)>=8:result+=w[:,None]*(x[6]*np.array(fit['local_tangent_u'])+x[7]*np.array(fit['local_tangent_v']))
        if len(x)>=9:
            radial=v-np.array(fit['local_bowl_center']);radial-=(radial@axis)[:,None]*axis
            result+=(1-w[:,None])*(x[8]-1)*radial
        return result
    assets=root.find('asset');contact=root.find('contact');new_parts={};changes=[]
    for o in poses:
        body=root.find(f".//body[@name='{o}']");body.set('pos',values(poses[o][:3]));body.set('quat',values(poses[o][3:]))
        visual=assets.find(f"mesh[@name='{o}_visual_0']")
        if o=='white_pot':visual.set('file',str(OUT/version/'white_pot_visual.obj'))
        if mode=='warped32':
            paths=sorted((B.parent/f'final_recon/episode_000001/assets/{o}_convex').glob('*.obj'))
            if o=='white_pot':
                dest=out/'pot_convex';dest.mkdir(exist_ok=True);new=[]
                for path in paths:
                    mesh=trimesh.load(path,force='mesh',process=False);mesh.vertices=warp(mesh.vertices)
                    result=mesh.convex_hull;target=dest/path.name;result.export(target);new.append(target)
                paths=new
        elif mode=='patch':
            patch_dir=OUT/('patch_collision_'+version)/o
            report=json.loads((patch_dir/'report.json').read_text())
            visual_source=OUT/version/'white_pot_visual.obj' if o=='white_pot' else B.parent/'final_recon/episode_000001/assets/white_pot_lid_visual.obj'
            if hashlib.sha256(visual_source.read_bytes()).hexdigest()!=report['source_sha256']:raise ValueError('Patch collision source hash mismatch')
            paths=sorted(patch_dir.glob('part_*.obj'))
        else:
            fine_dir=OUT/('fine_collision'+suffix)/o
            # The lid mesh is immutable in these shape variants; its existing
            # fine decomposition may be reused only after verifying that hash.
            if o=='white_pot_lid' and not fine_dir.exists():
                fine_dir=OUT/'fine_collision'/o
                report=json.loads((fine_dir/'report.json').read_text())
                actual=hashlib.sha256((B.parent/'final_recon/episode_000001/assets/white_pot_lid_visual.obj').read_bytes()).hexdigest()
                if actual!=report['source_sha256']:raise ValueError('Lid decomposition source hash changed')
            paths=sorted(fine_dir.glob('part_*.obj'))
            if not paths:raise FileNotFoundError(f'fine collision not ready: {o}')
        new_parts[o]=[str(p) for p in paths]
        old_geoms=[g for g in body.findall('geom') if '_collision_' in g.get('name','')]
        template=copy.deepcopy(old_geoms[0]);old_names={g.get('name') for g in old_geoms}
        old_meshes={g.get('mesh') for g in old_geoms}
        for g in old_geoms:body.remove(g)
        for mesh in list(assets):
            if mesh.tag=='mesh' and mesh.get('name') in old_meshes:assets.remove(mesh)
        for pair in list(contact):
            if pair.get('geom1') in old_names or pair.get('geom2') in old_names:contact.remove(pair)
        for i,path in enumerate(paths):
            name=f'{o}_collision_{i}';ET.SubElement(assets,'mesh',name=name,file=str(path.resolve()))
            g=copy.deepcopy(template);g.set('name',name);g.set('mesh',name);body.append(g)
    # Serialize the same hand initial state with the explicitly repaired object
    # states. This is one deterministic registered scene, never per-trial starts.
    oldmodel=mujoco.MjModel.from_xml_path(str(SCENE));q=oldmodel.key_qpos[0].copy()
    for o,p in poses.items():
        a=int(oldmodel.joint(o+'_free').qposadr[0]);q[a:a+7]=p
    root.find('keyframe/key').set('qpos',values(q))
    ET.indent(root);scene=out/'scene_repaired.xml';ET.ElementTree(root).write(scene,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
    relevant=[]
    for c in data.contact:
        names=[model.body(int(model.geom_bodyid[g])).name for g in (c.geom1,c.geom2)]
        if any(o in names for o in poses):relevant.append(dict(bodies=names,distance_m=float(c.dist)))
    provenance=dict(mode=mode,original_scene=str(SCENE),scene=str(scene),source_preserved=True,visual_mesh_changed=['white_pot'],
        geometry_fit=str(OUT/version/'fit.json'),support_source=str(B/'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json'),
        original_proxy_targets_unchanged=True,fixed_reconstructed_initial_poses={k:v.tolist() for k,v in poses.items()},
        gravity_setup=(-9.81*n).tolist(),normal_setup=n.tolist(),plane_offset_setup=offset,collision_parts=new_parts,
        masses_and_inertias='original source values retained for controlled collision/registration comparison',
        initial_object_contacts=relevant,initial_max_penetration_m=max([max(0.,-r['distance_m']) for r in relevant]+[0.]))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    return scene,out,provenance


def run(mode,version='pot_geometry_rgb_depth'):
    scene,out,provenance=prepare(mode,version)
    model=mujoco.MjModel.from_xml_path(str(scene));q=model.key_qpos[0].copy();reference={'qpos':np.tile(q,(41,1)),'fps':20.}
    execution=execute_reference(scene,reference,out/'passive')
    frozen=mujoco.MjModel.from_xml_path(execution['scene']);d=mujoco.MjData(frozen);mujoco.mj_resetDataKeyframe(frozen,d,0)
    z=np.load(execution['trajectory']);force=np.zeros(6);trace=[]
    for step,ctrl in enumerate(z['substep_ctrl']):
        d.ctrl[:]=ctrl;mujoco.mj_step(frozen,d)
        if step%25:continue
        total=0.;penetration=0.
        for i,c in enumerate(d.contact):
            bodies={frozen.body(int(frozen.geom_bodyid[g])).name for g in (c.geom1,c.geom2)}
            if bodies=={'white_pot','white_pot_lid'}:
                mujoco.mj_contactForce(frozen,d,i,force);total+=max(0.,float(force[0]));penetration=max(penetration,max(0.,-float(c.dist)))
        trace.append(dict(time_s=float(d.time),pot_lid_normal_force_n=total,pot_lid_max_penetration_m=penetration))
    movement={}
    for o in ['white_pot','white_pot_lid']:
        a=int(frozen.joint(o+'_free').qposadr[0]);start=z['qpos'][0,a:a+7];end=z['qpos'][-1,a:a+7]
        movement[o]=dict(translation_m=float(np.linalg.norm(end[:3]-start[:3])),rotation_deg=float(np.rad2deg((Rotation.from_quat(end[3:],scalar_first=True)*Rotation.from_quat(start[3:],scalar_first=True).inv()).magnitude())))
    report=dict(provenance=provenance,execution=execution,passive_object_drift=movement,force_trace=trace)
    (out/'passive_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(mode=mode,physics_verified=execution['physics_verified'],initial_penetration_m=provenance['initial_max_penetration_m'],movement=movement,final_force=trace[-1]),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['warped32','fine','patch'],default='warped32')
    p.add_argument('--version',default='pot_geometry_rgb_depth');run(**vars(p.parse_args()))
