"""Fit a small pot-shape repair to RGB/stereo only, never to task success.

The fitted degrees of freedom are bowl height and smooth handle elevation/bend.
No source files are changed. A proposed mesh is exported only as a new candidate.
"""
from pathlib import Path
import argparse,json
import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,EP,OUT,load,outline,mask,dominant_normal


def run(max_evaluations=900):
    m,d,camera,meshes=load();mesh=meshes['white_pot'];source=cv2.imread(str(OUT/'source_000.jpg'))
    folder=OUT/'pot_geometry_rgb_depth';folder.mkdir(exist_ok=True)
    support=json.loads((B/'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json').read_text())
    T=np.array(support['proposed_setup_from_source']);n=T[2,:3];shift=np.array(json.loads((EP/'input_manifest.json').read_text())['setup_from_source'])[:3,3]
    offset=T[2,3]-n@shift;tangent=T[:2,:3];axis=dominant_normal(mesh,bottom=True)
    local_floor=float((mesh.vertices@axis).min());base=np.array(json.loads((OUT/'pot_table_candidate.json').read_text())['candidate_pot_pose'])
    raw=np.load(OUT/'white_pot_depth_points.npz');pc=raw['points_source']+shift;h=pc@n+offset
    keep=(h>.015)&(h<.16);pc=pc[keep];pixels=raw['pixels'][keep]
    # Equalize body and handle evidence so the long visible handle cannot cause
    # the old failure mode of tipping the entire pot to fit the handle alone.
    groups=[np.flatnonzero(pixels[:,1]<1200),np.flatnonzero(pixels[:,1]>=1200)]
    selected=np.concatenate([g[::max(1,len(g)//700)] for g in groups]);observed=pc[selected]
    original_pose=np.r_[d.xpos[m.body('white_pot').id],d.xquat[m.body('white_pot').id]]
    lid_pose=np.r_[d.xpos[m.body('white_pot_lid').id],d.xquat[m.body('white_pot_lid').id]]
    sampled,_=trimesh.sample.sample_surface(mesh,110000,seed=904)
    cfg=json.loads((B.parent/'final_recon/episode_000001/config.json').read_text())
    small_camera={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in camera.items()};small_camera['K']=camera['K'].copy();small_camera['K'][:2]/=3
    shape=(source.shape[0]//3,source.shape[1]//3,3)
    truth=cv2.resize(cv2.imread(str(Path(cfg['object_masks']['white_pot'])/'000000.png'),0),(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)>0
    lidmask=mask(meshes['white_pot_lid'],lid_pose,small_camera,shape)>0

    def warp(vertices,x):
        height=vertices@axis-local_floor
        w=np.clip((vertices[:,2]-.075)/.055,0.,1.);w=w*w*(3-2*w)
        displacement=(x[3]-1)*height*(1-w)+w*(x[4]+x[5]*(vertices[:,2]-.13))
        return vertices+displacement[:,None]*axis

    def geometry(x):
        proposed=mesh.copy();proposed.vertices=warp(mesh.vertices,x)
        pose=base.copy();pose[:3]+=x[:2]@tangent
        rotation=Rotation.from_rotvec(n*x[2])*Rotation.from_quat(pose[3:],scalar_first=True);pose[3:]=rotation.as_quat(scalar_first=True)
        floor=np.min((proposed.vertices@rotation.as_matrix().T+pose[:3])@n+offset)
        pose[:3]+=(.0001-floor)*n
        return proposed,pose

    def stats(x):
        proposed,pose=geometry(x);R=Rotation.from_quat(pose[3:],scalar_first=True).as_matrix()
        local=(observed-pose[:3])@R;dist=cKDTree(warp(sampled,x)).query(local)[0]
        pred=(mask(proposed,pose,small_camera,shape)>0)&~lidmask
        return dict(mean_stereo_surface_distance_m=float(np.minimum(dist,.03).mean()),median_stereo_surface_distance_m=float(np.median(dist)),
            p90_stereo_surface_distance_m=float(np.quantile(dist,.9)),visible_mask_iou=float((pred&truth).sum()/max(1,(pred|truth).sum())))
    history=[]
    def objective(x):
        ss=stats(x);cost=800*ss['mean_stereo_surface_distance_m']+14*(1-ss['visible_mask_iou'])
        # Weak shape prior favors the smallest deformation supported by data.
        cost+=.3*((x[3]-1)/.15)**2+.05*(x[4]/.03)**2+.05*(x[5]/.3)**2
        history.append(dict(cost=cost,parameters=x.tolist(),stats=ss))
        if len(history)%25==0:print('eval',len(history),'best',min(p['cost'] for p in history),flush=True)
        return cost
    initial=np.array([0.,0.,0.,1.,0.,0.]);before=stats(initial)
    fit=minimize(objective,initial,method='Powell',bounds=[(-.025,.025),(-.025,.025),(-.12,.12),(.70,1.15),(-.01,.055),(-.35,.5)],
        options=dict(maxiter=10,maxfev=max_evaluations,xtol=1e-4,ftol=2e-5))
    best=min(history,key=lambda r:r['cost']);x=np.array(best['parameters']);proposed,pose=geometry(x)
    proposed.export(folder/'white_pot_visual.obj')
    panels=[]
    for title,display_mesh,p in [('Original source pose/mesh',mesh,original_pose),('Rigid pot on measured table',mesh,base),('RGB/stereo-fitted pot shape candidate',proposed,pose)]:
        panel=source.copy();outline(panel,display_mesh,p,camera,(255,150,0));outline(panel,meshes['white_pot_lid'],lid_pose,camera,(0,0,255))
        cv2.putText(panel,title,(650,740),cv2.FONT_HERSHEY_SIMPLEX,.63,(0,255,255),2);panels.append(cv2.resize(panel[700:1520,650:1450],(640,656)))
    cv2.imwrite(str(folder/'rgb_overlay.jpg'),np.hstack(panels))
    result=dict(provisional=True,ranking_data='Frame0 original RGB masks and calibrated stereo depth only; no physics/task reward',
        shape_model='bowl height scaling + smooth elevation/bending of handle along mesh bottom-plane normal',
        local_height_axis=axis.tolist(),local_floor=local_floor,parameters=x.tolist(),parameter_names=['tangent_x_m','tangent_y_m','yaw_rad','bowl_height_scale','handle_elevation_m','handle_slope'],
        pose=pose.tolist(),unchanged_lid_pose=lid_pose.tolist(),rigid_table_aligned_stats=before,candidate_stats=best['stats'],
        optimizer_success=bool(fit.success),evaluations=len(history),original_inputs_unchanged=True,collision_meshes_not_yet_changed=True)
    (folder/'fit.json').write_text(json.dumps(result,indent=2));(folder/'history.json').write_text(json.dumps(history))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--max-evaluations',type=int,default=900);run(parser.parse_args().max_evaluations)
