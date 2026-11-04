"""Direct convex patches of reconstructed thin-shell triangles.

This avoids watertight volume inference by CoACD's preprocessing. Every original
triangle belongs to exactly one patch, whose convex hull contains that triangle.
No object visual, initial pose, mass, or contact parameter is changed here.
The grid and any numerical coplanar thickness are recorded explicitly.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np,trimesh
from scipy.spatial import ConvexHull,QhullError,cKDTree
from repair_episode1_registration import B,OUT,dominant_normal

def run(version='pot_geometry_rgb_depth_height094',cell=.025):
    folder=OUT/('patch_collision_'+version);folder.mkdir(exist_ok=True)
    reports=[]
    for name in ['white_pot','white_pot_lid']:
        source=OUT/version/'white_pot_visual.obj' if name=='white_pot' else B.parent/'final_recon/episode_000001/assets/white_pot_lid_visual.obj'
        mesh=trimesh.load(source,force='mesh',process=False)
        # Split long triangles first; a face crossing many cells would otherwise
        # make a large convex shortcut even with a fine centroid partition.
        vertices,faces=trimesh.remesh.subdivide_to_size(mesh.vertices,mesh.faces,max_edge=cell*.65,max_iter=5)
        triangles=vertices[faces];axis=dominant_normal(mesh,bottom=name=='white_pot');u=np.array([0.,1.,0.]);u-=u@axis*axis;u/=np.linalg.norm(u);v=np.cross(axis,u);basis=np.stack([u,v,axis])
        centers=triangles.mean(1)@basis.T;keys=np.floor((centers-mesh.vertices.mean(0)@basis.T)/cell).astype(int)
        unique,inverse=np.unique(keys,axis=0,return_inverse=True);dest=folder/name;dest.mkdir(exist_ok=True)
        if (dest/'report.json').exists():raise FileExistsError('Preserve completed patches: '+str(dest))
        original_surface,_=trimesh.sample.sample_surface(mesh,250000,seed=661);tree=cKDTree(original_surface)
        parts=[];distances=[];volume=0.;coplanar=0
        for k,key in enumerate(unique):
            points=np.unique(triangles[inverse==k].reshape(-1,3),axis=0);center=points.mean(0);_,s,vh=np.linalg.svd(points-center,full_matrices=False)
            if len(points)<4 or s[-1]<1e-8:
                # A zero-thickness triangle is not a valid MuJoCo mesh volume.
                # Add 0.1mm per side only for genuinely coplanar patches.
                normal=vh[-1];points=np.r_[points+normal*.0001,points-normal*.0001];coplanar+=1
            hull=ConvexHull(points);used=np.unique(hull.simplices);part=trimesh.Trimesh(points,hull.simplices,process=True).convex_hull
            path=dest/f'part_{k:03d}.obj';part.export(path);parts.append(str(path));volume+=part.volume
            sampled,_=trimesh.sample.sample_surface(part,100,seed=900+k);distances.extend(tree.query(sampled)[0])
        report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),parts=len(parts),files=parts,cell_m=cell,
            method='All source triangles partitioned by local centroid grid, convex hull per patch; no inferred watertight volume',
            subdivided_triangles=len(triangles),source_triangles=len(mesh.faces),coplanar_patches_with_0p2mm_thickness=coplanar,
            sum_convex_volumes_m3=float(volume),sampled_unsigned_surface_deviation_m=np.quantile(distances,[.5,.9,.99,1]).tolist(),
            approximation_warning='Nearest sampled source-surface distances include sampling error and do not certify signed-volume accuracy; full physics still required.')
        (dest/'report.json').write_text(json.dumps(report,indent=2));reports.append(report);print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)
    (folder/'report.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',default='pot_geometry_rgb_depth_height094');p.add_argument('--cell',type=float,default=.025);run(**vars(p.parse_args()))
