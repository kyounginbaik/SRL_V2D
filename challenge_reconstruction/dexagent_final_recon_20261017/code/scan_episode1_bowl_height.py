"""Audit a physically constrained bowl-height completion against RGB-D.

The lid registration and all observations remain immutable. This is a shape
reconstruction diagnostic, never a success-oriented simulation parameter sweep.
"""
from pathlib import Path
import copy,json
import cv2,mujoco,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,EP,OUT,outline
from register_episode1_multiframe import rigid_fit
from validate_episode1_repair import prepare

def run():
    src=B.parent/'final_recon/episode_000001';basefit=json.loads((OUT/'multiframe_shape_v1/fit.json').read_text())
    mesh=trimesh.load(src/'assets/white_pot_visual.obj',force='mesh',process=False);sample,_=trimesh.sample.sample_surface(mesh,110000,seed=771)
    a=np.array(basefit['local_height_axis']);floor=basefit['local_floor'];u=np.array(basefit['local_tangent_u']);v=np.array(basefit['local_tangent_v']);c=np.array(basefit['local_bowl_center'])
    shift=np.array(json.loads((EP/'input_manifest.json').read_text())['setup_from_source'])[:3,3]
    support=json.loads((B/'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json').read_text());table=np.array(support['proposed_setup_from_source']);n=table[2,:3]
    def warp(V,x):
        h=V@a-floor;w=np.clip((V[:,2]-.075)/.055,0,1);w=w*w*(3-2*w);r=V-c;r-=(r@a)[:,None]*a
        return V+((x[3]-1)*h*(1-w)+w*(x[4]+x[5]*(V[:,2]-.13)))[:,None]*a+w[:,None]*(x[6]*u+x[7]*v)+(1-w[:,None])*(x[8]-1)*r
    output=OUT/'bowl_height_audit';output.mkdir(exist_ok=True);reports=[];panels=[]
    for height in [.82,.86,.90,.94,.98,1.02]:
        name=f'pot_geometry_rgb_depth_height{round(height*100):03d}';folder=OUT/name;folder.mkdir(exist_ok=True)
        fit=copy.deepcopy(basefit);x=np.array(fit['parameters']);x[3]=height;candidate=mesh.copy();candidate.vertices=warp(mesh.vertices,x);candidate.export(folder/'white_pot_visual.obj')
        pose=np.array(fit['pose']);R=Rotation.from_quat(pose[3:],scalar_first=True).as_matrix();low=((candidate.vertices@R.T+pose[:3]-shift)@n+table[2,3]).min();pose[:3]+=(.0001-low)*n
        fit.update(parameters=x.tolist(),pose=pose.tolist(),ranking_data='Bounded height-completion audit with independently measured tabletop and fixed observed lid; not promoted',provisional=True)
        (folder/'fit.json').write_text(json.dumps(fit,indent=2));scene,out,provenance=prepare('warped32',name)
        model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,data,0);mujoco.mj_forward(model,data)
        pen=max([max(0.,-float(q.dist)) for q in data.contact if {model.body(int(model.geom_bodyid[g])).name for g in (q.geom1,q.geom2)}=={'white_pot','white_pot_lid'}]+[0.])
        pts=warp(sample,x);tree=cKDTree(pts);record=dict(height_scale=height,version=name,initial_pot_lid_penetration_m=pen,frames=[])
        for original in basefit['frames']:
            frame=original['frame'];obs=np.load(OUT/f'multiframe_clouds/white_pot_{frame:06d}.npz');pc=obs['points_source'];pc=pc[np.linspace(0,len(pc)-1,min(2200,len(pc))).astype(int)]
            T=np.array(original['world_from_object'])
            if frame==0:T[:3,:3]=R;T[:3,3]=pose[:3]-shift
            else:
                for _ in range(35):
                    dist,idx=tree.query((pc-T[:3,3])@T[:3,:3]);keep=dist<min(.025,np.quantile(dist,.88))
                    RR,pp=rigid_fit(pts[idx[keep]],pc[keep]);T[:3,:3]=RR;T[:3,3]=pp
            dist=tree.query((pc-T[:3,3])@T[:3,:3])[0];record['frames'].append(dict(frame=frame,median_m=float(np.median(dist)),p80_m=float(np.quantile(dist,.8)),world_from_object=T.tolist()))
            if frame in [0,120]:
                rgb=cv2.imread(str(OUT/f'multiframe_clouds/rgb_{frame:06d}.jpg'));cam={'K':obs['K'],'world_from_rgb':obs['world_from_rgb'][None]};p=np.r_[T[:3,3],Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True)]
                outline(rgb,candidate,p,cam,(0,255,0));cv2.putText(rgb,f'Bowl height {height:.2f}, t={frame/20:g}s',(600,740),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
                panels.append(cv2.resize(rgb[700:1450,550:1500],(570,450)))
        reports.append(record);print(json.dumps(record),flush=True);(output/'report.json').write_text(json.dumps(reports,indent=2))
    cv2.imwrite(str(output/'overlay.jpg'),np.vstack([np.hstack(panels[i:i+2]) for i in range(0,len(panels),2)]))

if __name__=='__main__':run()
