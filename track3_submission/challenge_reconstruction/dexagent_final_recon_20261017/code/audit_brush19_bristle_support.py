"""Source-supported bristle-envelope candidates; never edit original assets/GT.

The visible bristles rest on the measured table. The rigid reconstruction has a
slanted, ragged lower envelope and topples before either hand touches it. Fit
candidate lower geometry, quantify RGB/stereo agreement, then test unchanged
commands in a versioned physical scene. These are provisional reconstruction
experiments, not accepted demonstrations or submission assets.
"""
from pathlib import Path
import argparse,json,sys,xml.etree.ElementTree as ET
import cv2,mujoco,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from repair_episode1_registration import mask,outline
from floating_runtime import execute_reference
from evaluate_mp_sr import evaluate_files

B=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(B.parents[1]/'third_party/DexAgent'))

def run(alpha):
    source=B.parent/'final_recon/episode_000019';inp=B/'episodes/episode_000019'
    baseline=B/'refinement_20261018/expanded_batch/native_mpc/episode_000019';old=json.loads((baseline/'result.json').read_text())
    manifest=json.loads((inp/'input_manifest.json').read_text());A=np.array(manifest['setup_from_source']);poses=np.load(source/'object_poses.npz');slot=list(poses['names']).index('mini_sweeper');T0=A@poses['world_from_object'][0,slot]
    original=trimesh.load(source/'assets/mini_sweeper_visual.obj',process=False);world=original.vertices@T0[:3,:3].T+T0[:3,3]
    centers=original.triangles_center@T0[:3,:3].T+T0[:3,3];normals=original.face_normals@T0[:3,:3].T
    low=(centers[:,2]<.02)&(normals[:,2]<-.3)
    plane=np.linalg.lstsq(np.c_[centers[low,:2],np.ones(low.sum())],centers[low,2],rcond=None)[0]
    def deform(vertices):
        w=vertices@T0[:3,:3].T+T0[:3,3];base=np.c_[w[:,:2],np.ones(len(w))]@plane
        h=.045;selected=w[:,2]<h;z=np.maximum(0.,h*(w[:,2]-base)/np.maximum(.015,h-base))
        w[selected,2]=(1-alpha)*w[selected,2]+alpha*z[selected]
        return (w-T0[:3,3])@T0[:3,:3]
    out=B/f'refinement_20261018/expanded_batch/brush19_support/alpha_{alpha:.2f}';out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise FileExistsError(out)
    candidate=original.copy();candidate.vertices=deform(original.vertices);candidate.export(out/'mini_sweeper_visual.obj')
    camera=dict(np.load(inp/'camera_trajectory.npz'));clouds=B/'refinement_20261018/expanded_batch/source_review/episode_000019/stereo_clouds'
    config=json.loads((source/'config.json').read_text());rows=[];panels=[]
    sampled,_=trimesh.sample.sample_surface(candidate,70000,seed=19);sampled0,_=trimesh.sample.sample_surface(original,70000,seed=19)
    for frame in [0,5,10,15]:
        z=np.load(clouds/f'mini_sweeper_{frame:06d}.npz');T=z['world_from_object'];points=(z['points_source']-T[:3,3])@T[:3,:3]
        common=points@T0[:3,:3].T+T0[:3,3];hsv=cv2.cvtColor(z['rgb'].reshape(-1,1,3),cv2.COLOR_BGR2HSV).reshape(-1,3)
        bristles=(common[:,2]<.045)&(common[:,2]>-.01)&(hsv[:,1]<110)&(hsv[:,2]>60);points=points[bristles]
        rgb=cv2.imread(str(clouds/f'rgb_{frame:06d}.jpg'));H,W=rgb.shape[:2];observed=cv2.imread(str(Path(config['object_masks']['mini_sweeper'])/f'{frame:06d}.png'),0);observed=cv2.resize(observed,(W,H),interpolation=cv2.INTER_NEAREST)>0
        C=dict(camera,world_from_rgb=camera['world_from_rgb'][frame:frame+1]);pose=A@poses['world_from_object'][frame,slot];pose=np.r_[pose[:3,3],Rotation.from_matrix(pose[:3,:3]).as_quat(scalar_first=True)]
        metrics={}
        for label,mesh,samples,color in [('original',original,sampled0,(0,0,255)),('candidate',candidate,sampled,(0,255,0))]:
            dist=cKDTree(samples).query(points)[0];projected=mask(mesh,pose,C,rgb.shape)>0
            metrics[label]=dict(bristle_surface_p50_m=float(np.median(dist)),bristle_surface_p80_m=float(np.quantile(dist,.8)),mask_iou=float((projected&observed).sum()/max(1,(projected|observed).sum())))
        rows.append(dict(frame=frame,points=len(points),**metrics))
        if frame in [0,10]:
            panel=rgb.copy();outline(panel,original,pose,C,(0,0,255));outline(panel,candidate,pose,C,(0,255,0));ys,xs=np.where(observed)
            crop=panel[max(0,ys.min()-70):min(H,ys.max()+70),max(0,xs.min()-70):min(W,xs.max()+70)]
            panels.append(cv2.resize(crop,(600,600)))
    cv2.imwrite(str(out/'source_overlay_red_original_green_candidate.jpg'),np.hstack(panels))
    (out/'source_evidence.json').write_text(json.dumps(rows,indent=2)+'\n')
    tree=ET.parse(old['execution']['scene']);changed=[]
    for element in tree.findall('.//asset/mesh'):
        if not element.get('name','').startswith('mini_sweeper_'):continue
        original_path=Path(element.get('file'));mesh=trimesh.load(original_path,process=False);mesh.vertices=deform(mesh.vertices)
        target=out/'assets'/original_path.name;target.parent.mkdir(exist_ok=True);mesh.export(target);element.set('file',str(target.resolve()));changed.append(str(target))
    scene=out/'scene.xml';tree.write(scene,encoding='unicode')
    reference=Path(old['search']['reference']);execution=execute_reference(scene,reference,out/'execution')
    metric=evaluate_files(manifest['source'],execution['trajectory'],inp/'input_manifest.json',execution['scene'],inp/'metric_geometry.json')
    with np.load(execution['trajectory']) as z:q=z['qpos']
    m=mujoco.MjModel.from_xml_path(execution['scene']);adr=m.joint('mini_sweeper_free').qposadr[0]
    angle=np.degrees((Rotation.from_quat(q[:,adr+3:adr+7],scalar_first=True)*Rotation.from_quat(q[0,adr+3:adr+7],scalar_first=True).inv()).magnitude())
    report=dict(episode='episode_000019',provisional_reconstruction_candidate=True,alpha=alpha,mesh_lower_face_plane=plane.tolist(),
        geometry_method='Warp only geometry below4.5cm toward the measured tabletop bristle envelope; original upper brush/handle fixed',
        source_poses_unchanged=True,original_files_unchanged=True,robot_commands_unchanged=True,physical_mass_inertia_gains_limits_unchanged=True,
        source_evidence=rows,changed_meshes=changed,execution=execution,metrics_with_original_geometry_anchor=metric,
        brush_initial_tilt_at_point6_s_degrees=float(angle[12]),submission_ready=False,
        limitations=['Rigid-envelope approximation of compliant bristles is provisional.','Candidate visual and collision geometry require source review before adoption.','Original GT and original geometry anchor remain unchanged for comparison; no pose alignment perobject is used.'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(output=str(out),evidence=rows,initial_tilt_deg=float(angle[12]))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--alpha',type=float,required=True);a=p.parse_args();run(a.alpha)
