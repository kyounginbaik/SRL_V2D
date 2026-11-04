"""Repair proven non-manifold source-lid topology in versioned candidates.

No task score, grasp information, or physical contact target enters this repair.
The visible stereo fit and bidirectional surface displacement are reported.
"""
import argparse,hashlib,json,time
import cv2,numpy as np,trimesh,pymeshfix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,EP,OUT,load,outline,mask

def topology(mesh):
    _,f=np.unique(np.sort(mesh.faces,axis=1),axis=0,return_counts=True);_,e=np.unique(mesh.edges_sorted,axis=0,return_counts=True)
    return dict(vertices=len(mesh.vertices),faces=len(mesh.faces),duplicate_faces=int((f-1).sum()),zero_area_faces=int((mesh.area_faces<1e-12).sum()),
        boundary_edges=int((e==1).sum()),nonmanifold_edges=int((e>2).sum()),watertight=bool(mesh.is_watertight),winding_consistent=bool(mesh.is_winding_consistent),signed_volume_m3=float(mesh.volume))

def run(join=False):
    label='meshfix_join' if join else 'meshfix_keep';folder=OUT/label;folder.mkdir(exist_ok=True)
    if (folder/'report.json').exists():raise FileExistsError(folder)
    src=B.parent/'final_recon/episode_000001/assets/white_pot_lid_visual.obj';mesh=trimesh.load(src,force='mesh',process=False)
    start=time.monotonic();vertices,faces=pymeshfix.clean_from_arrays(mesh.vertices,np.asarray(mesh.faces,dtype=np.int32),verbose=True,joincomp=join,remove_smallest_components=False)
    repaired=trimesh.Trimesh(vertices,faces,process=False);trimesh.repair.fix_normals(repaired,multibody=True);repaired.export(folder/'white_pot_lid_visual.obj')
    original_sample,_=trimesh.sample.sample_surface(mesh,250000,seed=311);fixed_sample,_=trimesh.sample.sample_surface(repaired,250000,seed=312)
    a=cKDTree(original_sample);b=cKDTree(fixed_sample);forward=b.query(original_sample)[0];backward=a.query(fixed_sample)[0]
    m,d,camera,_=load();pose=np.r_[d.xpos[m.body('white_pot_lid').id],d.xquat[m.body('white_pot_lid').id]];R=Rotation.from_quat(pose[3:],scalar_first=True).as_matrix()
    shift=np.array(json.loads((EP/'input_manifest.json').read_text())['setup_from_source'])[:3,3]
    pc=np.load(OUT/'white_pot_lid_depth_points.npz')['points_source']+shift;dist0=a.query((pc-pose[:3])@R)[0];dist1=b.query((pc-pose[:3])@R)[0]
    rgb=cv2.imread(str(OUT/'source_000.jpg'));panels=[]
    for title,mm,color in [('Source invalid topology',mesh,(255,150,0)),(label,repaired,(0,255,0))]:
        panel=rgb.copy();outline(panel,mm,pose,camera,color);cv2.putText(panel,title,(650,740),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,255,255),2);panels.append(cv2.resize(panel[700:1250,650:1450],(640,440)))
    cv2.imwrite(str(folder/'overlay.jpg'),np.hstack(panels))
    result=dict(source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),source_preserved=True,method='PyMeshFix0.18.1 clean_from_arrays',
        parameters=dict(joincomp=join,remove_smallest_components=False),documentation='https://pymeshfix.pyvista.org/',
        source_topology=topology(mesh),candidate_topology=topology(repaired),source_to_candidate_distance_quantiles_m=np.quantile(forward,[.5,.9,.99,1]).tolist(),
        candidate_to_source_distance_quantiles_m=np.quantile(backward,[.5,.9,.99,1]).tolist(),
        initial_observed_stereo_median_m=dict(source=float(np.median(dist0)),candidate=float(np.median(dist1))),
        initial_observed_stereo_p90_m=dict(source=float(np.quantile(dist0,.9)),candidate=float(np.quantile(dist1,.9))),
        elapsed_s=time.monotonic()-start,provisional=True,physics_not_tested=True)
    (folder/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--join',action='store_true');run(**vars(p.parse_args()))
