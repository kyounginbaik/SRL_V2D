"""Read-only source topology and physical-bottom audit; never edits scenes."""
from pathlib import Path
import hashlib,json,xml.etree.ElementTree as ET
import numpy as np
import trimesh
import mujoco
from audit_cup_cavities import ray_convex,hull_equations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B=Path(__file__).resolve().parents[1]
SRC=B.parent/'final_recon/episode_000010/assets'
SCENE=B/'refinement_20261018/episode_000010/learned_feedback/gain_0.03/evaluation/execution/scene_physics.xml'
OUT=B/'refinement_20261018/episode_000010/learned_feedback/gain_0.03/audit/cup_bottoms'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hits(mesh,origin,direction):
    tri=mesh.triangles;e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    h=np.cross(direction,e2);a=np.einsum('ij,ij->i',e1,h);valid=np.abs(a)>1e-12
    f=np.zeros_like(a);f[valid]=1/a[valid];s=origin-tri[:,0];u=f*np.einsum('ij,ij->i',s,h)
    q=np.cross(s,e1);v=f*(q@direction);t=f*np.einsum('ij,ij->i',e2,q)
    ok=valid&(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(t>1e-8)
    idx=np.flatnonzero(ok);idx=idx[np.argsort(t[idx])]
    return [{'z_m':float((origin+t[k]*direction)[2]),'face':int(k),'normal_dot_ray':float(mesh.face_normals[k]@direction)} for k in idx]

def main():
    OUT.mkdir(parents=True,exist_ok=True);tree=ET.parse(SCENE);model=mujoco.MjModel.from_xml_path(str(SCENE))
    report={'scene':str(SCENE),'scene_sha256':sha(SCENE),'objects':{},'notes':['Rays start inside the open cup above its floor and point toward local +Z, which points toward the base in these original assets.','Two-sided triangle intersections establish whether geometry exists independent of backface culling. Convex-part intersections test the exact collision union used by MuJoCo.','Only diagnostics are written; passing scene and meshes are unchanged.']}
    fig,axs=plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
    coords=np.linspace(-.02,.02,41)
    for row,name in enumerate(('beige_cup','blue_cup')):
        path=SRC/f'{name}.obj';visual=SRC/f'{name}_visual.obj';mesh=trimesh.load(visual,force='mesh',process=False)
        edges=np.sort(mesh.edges,axis=1);_,counts=np.unique(edges,axis=0,return_counts=True)
        faces=np.sort(mesh.faces,axis=1);dups=len(faces)-len(np.unique(faces,axis=0))
        source_vertices=mesh.vertices.copy()
        eq=[];files=[]
        for node in tree.findall('.//asset/mesh'):
            if node.get('name','').startswith(name+'_collision_'):
                p=Path(node.get('file'));files.append(p);eq.append(hull_equations(trimesh.load(p,force='mesh',process=False)))
        gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name+'_visual')
        if gid<0:
            gid=next(i for i in range(model.ngeom) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(name) and model.geom_contype[i]==0 and model.geom_dataid[i]>=0)
        rgba=model.geom_rgba[gid].tolist();mat=int(model.geom_matid[gid])
        visualhit=np.zeros((len(coords),len(coords)));fronthit=visualhit.copy();collisionhit=visualhit.copy();zmin=float(mesh.bounds[0,2]-.02)
        samples=[]
        for iy,y in enumerate(coords):
            for ix,x in enumerate(coords):
                origin=np.array([x,y,zmin]);hs=hits(mesh,origin,np.array([0.,0.,1.]));rc=ray_convex(eq,origin,np.array([0.,0.,1.]))
                visualhit[iy,ix]=bool(hs);fronthit[iy,ix]=any(h['normal_dot_ray']<0 for h in hs);collisionhit[iy,ix]=rc is not None
                if x==0 and y==0:samples=hs
        r={'source_obj':str(path),'source_obj_sha256':sha(path),'visual_obj_sha256':sha(visual),'source_and_visual_identical':sha(path)==sha(visual),'vertices':len(mesh.vertices),'faces':len(mesh.faces),'duplicate_faces':dups,'degenerate_faces':int(np.sum(mesh.area_faces<1e-14)),'boundary_edges':int(np.sum(counts==1)),'nonmanifold_edges':int(np.sum(counts>2)),'edge_incidence_histogram':{str(k):int(np.sum(counts==k)) for k in np.unique(counts)},'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),'bounds_m':mesh.bounds.tolist(),'central_rays':{'grid_width_m':.04,'grid_points':len(coords)**2,'two_sided_visual_hit_fraction':float(visualhit.mean()),'front_facing_visual_hit_fraction':float(fronthit.mean()),'collision_union_hit_fraction':float(collisionhit.mean()),'center_visual_hits':samples,'center_collision_first_hit_z_m':None if (rc:=ray_convex(eq,np.array([0.,0.,zmin]),np.array([0.,0.,1.]))) is None else zmin+rc},'compiled_visual_geom':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,gid),'geom_rgba':rgba,'material_rgba':None if mat<0 else model.mat_rgba[mat].tolist(),'collision_parts':len(files),'collision_files_sha256':{str(p):sha(p) for p in files}}
        report['objects'][name]=r
        for col,(a,title) in enumerate(((visualhit,'two-sided visual floor'),(fronthit,'front-facing visual floor'),(collisionhit,'physical collision floor'))):
            axs[row,col].imshow(a,origin='lower',extent=[-20,20,-20,20],vmin=0,vmax=1,cmap='RdYlGn');axs[row,col].set(title=f'{name}: {title}\n{a.mean()*100:.1f}% of central 40 mm square',xlabel='local x (mm)',ylabel='local y (mm)')
    fig.savefig(OUT/'floor_ray_coverage.png',dpi=140);plt.close(fig)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in r.items() if k!='collision_files_sha256'} for name,r in report['objects'].items()},indent=2))

if __name__=='__main__':main()
