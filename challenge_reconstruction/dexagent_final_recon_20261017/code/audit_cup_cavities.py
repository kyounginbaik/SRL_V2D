#!/usr/bin/env python3
"""Read-only visual/collision cross-section audit for episode 10 cup nesting."""
from pathlib import Path
import json
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from grasp_stability import sha256

B=Path(__file__).resolve().parents[1]
OUT=B/'refinement_20261017/reference_collision_audit/episode10_cavities'

def hull_equations(mesh):return ConvexHull(mesh.vertices).equations

def ray_visual(mesh, origin, direction):
    tri=mesh.triangles;e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    h=np.cross(direction,e2);a=np.einsum('ij,ij->i',e1,h);valid=np.abs(a)>1e-12
    f=np.zeros_like(a);f[valid]=1/a[valid];s=origin-tri[:,0];u=f*np.einsum('ij,ij->i',s,h)
    q=np.cross(s,e1);v=f*(q@direction);t=f*np.einsum('ij,ij->i',e2,q)
    good=valid&(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(t>1e-8)
    return float(t[good].min()) if good.any() else None

def ray_convex(equations, origin, direction):
    hits=[]
    for eq in equations:
        a=eq[:,:3]@direction;b=-(eq[:,:3]@origin+eq[:,3]);positive=a>1e-12;negative=a<-1e-12
        if np.any((~positive&~negative)&(b<-1e-9)):continue
        lo=max(0.,float(np.max(b[negative]/a[negative])) if negative.any() else 0.)
        hi=float(np.min(b[positive]/a[positive])) if positive.any() else np.inf
        if lo<=hi and hi>=0:hits.append(lo)
    return min(hits) if hits else None

def section(mesh, normal, origin, u, v):
    seg=trimesh.intersections.mesh_plane(mesh,normal,origin)
    return np.stack([(seg-origin)@u,(seg-origin)@v],axis=-1)

def crossings(a,b):
    found=[]
    def cross(u,v):return u[...,0]*v[...,1]-u[...,1]*v[...,0]
    for x in a:
        p=x[0];r=x[1]-x[0];q=b[:,0];s=b[:,1]-b[:,0];den=cross(r,s);good=np.abs(den)>1e-12
        t=np.zeros(len(b));u=np.zeros(len(b));t[good]=cross(q-p,s)[good]/den[good];u[good]=cross(q-p,r)[good]/den[good]
        for k in np.flatnonzero(good&(t>1e-7)&(t<1-1e-7)&(u>1e-7)&(u<1-1e-7)):
            point=p+t[k]*r
            if not any(np.linalg.norm(point-other)<1e-4 for other in found):found.append(point)
    return np.asarray(found).reshape(-1,2)

def occupied(points,equations):
    flags=np.zeros(len(points),bool)
    for eq in equations:
        remaining=np.flatnonzero(~flags)
        flags[remaining]=np.all(points[remaining]@eq[:,:3].T+eq[:,3]<=1e-9,axis=1)
    return flags

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((B/'episodes/episode_000010/input_manifest.json').read_text())
    meshes={};collisions={};report={'episode':'episode_000010','assets':{},'cavity_rays':{},'nesting_sections':[]}
    for obj in manifest['objects']:
        name=obj['id'];visual=next(Path(x['source']) for x in obj['meshes'] if x['kind']=='visual');parts=[Path(x['source']) for x in obj['meshes'] if x['kind']=='collision']
        meshes[name]=trimesh.load(visual,force='mesh',process=False)
        collisions[name]=[trimesh.load(p,force='mesh',process=False).convex_hull for p in parts]
        report['assets'][name]={'visual_path':str(visual),'visual_sha256':sha256(visual),'watertight':bool(meshes[name].is_watertight),'winding_consistent':bool(meshes[name].is_winding_consistent),'collision_piece_count':len(parts),'collision_sha256':{str(p):sha256(p) for p in parts}}
        eq=[hull_equations(m) for m in collisions[name]];rows=[]
        for z in (-.06,-.03,0,.03,.06,.08):
            origin=np.array([0,0,z]);samples=[]
            for theta in np.linspace(0,2*np.pi,72,endpoint=False):
                d=np.array([np.cos(theta),np.sin(theta),0]);rv=ray_visual(meshes[name],origin,d);rc=ray_convex(eq,origin,d)
                samples.append({'angle_deg':float(np.degrees(theta)),'visual_first_hit_mm':None if rv is None else rv*1000,'collision_first_hit_mm':None if rc is None else rc*1000,'collision_inward_difference_mm':None if rv is None or rc is None else (rv-rc)*1000})
            diff=[s['collision_inward_difference_mm'] for s in samples if s['collision_inward_difference_mm'] is not None]
            rows.append({'local_z_mm':z*1000,'axis_point_inside_collider':bool(occupied(origin[None],eq)[0]),'max_collision_inward_difference_mm':max(diff),'mean_collision_inward_difference_mm':float(np.mean(diff)),'samples':samples})
        report['cavity_rays'][name]=rows
    fig,axs=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for row,(name,mesh) in enumerate(meshes.items()):
        for col,z in enumerate((-.06,0,.06)):
            ax=axs[row,col];origin=np.array([0,0,z]);u=np.array([1,0,0]);v=np.array([0,1,0]);normal=np.array([0,0,1]);s=section(mesh,normal,origin,u,v)
            for hull in collisions[name]:
                hs=section(hull,normal,origin,u,v)
                if len(hs):
                    pts=np.unique(hs.reshape(-1,2),axis=0)
                    if len(pts)>=3:
                        h=ConvexHull(pts);ax.fill(*(pts[h.vertices]*1000).T,color='tomato',alpha=.22,linewidth=0)
            ax.add_collection(LineCollection(s*1000,colors='black',linewidths=1.4));ax.set(xlim=(-55,55),ylim=(-55,55),aspect='equal',xlabel='local x (mm)',ylabel='local y (mm)',title=f'{name}: local z={z*1000:.0f} mm');ax.grid(alpha=.2)
    fig.suptitle('Cup cross-sections: black = visual surface; red = collision solid\nOriginal mesh coordinates; positive z points toward cup bottom in these assets')
    fig.savefig(OUT/'individual_cup_sections.png',dpi=160);plt.close(fig)
    with np.load(Path(manifest['source'])/'object_poses.npz',allow_pickle=False) as z:poses=z['world_from_object'];names=z['names'].astype(str).tolist()
    fig,axs=plt.subplots(2,2,figsize=(10,10),constrained_layout=True)
    for row,time in enumerate((2.8,8.0)):
        frame=round(time*manifest['fps']);relative=np.linalg.inv(poses[frame,names.index('blue_cup')])@poses[frame,names.index('beige_cup')]
        beige=meshes['beige_cup'].copy();beige.apply_transform(relative);parts=[h.copy() for h in collisions['beige_cup']]
        for h in parts:h.apply_transform(relative)
        u=relative[:3,3].copy();u[2]=0;u/=np.linalg.norm(u)
        for col,basis in enumerate((u,np.array([-u[1],u[0],0]))):
            ax=axs[row,col];v=np.array([0,0,1]);normal=np.cross(basis,v);origin=np.zeros(3)
            sections=[section(m,normal,origin,basis,v) for m in (meshes['blue_cup'],beige)]
            allpts=np.concatenate([s.reshape(-1,2) for s in sections]);lo=allpts.min(axis=0)-.01;hi=allpts.max(axis=0)+.01
            for j,(mesh,hs,color) in enumerate([(meshes['blue_cup'],collisions['blue_cup'],'royalblue'),(beige,parts,'darkorange')]):
                for h in hs:
                    s=section(h,normal,origin,basis,v)
                    if len(s):
                        pts=np.unique(s.reshape(-1,2),axis=0)
                        if len(pts)>=3:
                            hull=ConvexHull(pts);ax.fill(*(pts[hull.vertices]*1000).T,color=color,alpha=.18,linewidth=0)
                ax.add_collection(LineCollection(sections[j]*1000,colors=color,linewidths=1.5))
            crossing=crossings(*sections)
            if len(crossing):ax.scatter(*(crossing*1000).T,c='red',marker='x',s=30,zorder=5)
            report['nesting_sections'].append({'time_s':time,'plane':col,'basis_in_blue_frame':basis.tolist(),'beige_in_blue_transform':relative.tolist(),'visual_surface_crossings_count':len(crossing),'crossings_mm':(crossing*1000).tolist()})
            ax.set(xlim=(lo[0]*1000,hi[0]*1000),ylim=(lo[1]*1000,hi[1]*1000),aspect='equal',xlabel='transverse distance (mm)',ylabel='blue local z (mm)',title=f'{time:.2f}s, plane {col+1}: {len(crossing)} visual crossings');ax.invert_yaxis();ax.grid(alpha=.2)
    fig.suptitle('Unchanged nested references: visual contours plus translucent collision pieces\nBlue cup = blue; beige cup = orange; red x = intersecting visual surfaces')
    fig.savefig(OUT/'reference_nesting_sections.png',dpi=160);plt.close(fig)
    report['limitations']=['Radial cavity rays originate at local x=y=0; the first visual hit is used, not a watertight inside/outside classification.','Both visual meshes are open/non-watertight. Section boundary crossings prove intersecting surfaces in the chosen planes but do not define a reliable solid intersection volume.','Collision pieces are treated as their exact convex hulls, matching the experiment MuJoCo mesh contact model; assets and target poses are unchanged.','This is a diagnostic of these reconstructed assets/reference poses, not official hidden geometry.']
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('assets','cavity_rays')},indent=2))

if __name__=='__main__':main()
