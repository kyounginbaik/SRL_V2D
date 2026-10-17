"""Fit a single stable seated registration to stereo surfaces and source masks.

Uses fixed source meshes and a measured support normal; saves a separate proposal.
No simulation trajectories or original reference are overwritten.
"""
import json
from pathlib import Path
import cv2
import mujoco
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B, EP, OUT, load, outline, mask, dominant_normal


def run():
    m,d,camera,meshes=load()
    src=B.parent/'final_recon/episode_000001'
    cfg=json.loads((src/'config.json').read_text())
    support=json.loads((B/'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json').read_text())
    normal=np.array(support['fitted_plane_normal_source']);offset=float(support['proposed_setup_from_source'][2][3])
    shift=np.array(json.loads((EP/'input_manifest.json').read_text())['setup_from_source'])[:3,3]
    setup_offset=offset-normal@shift
    initial=json.loads((OUT/'pot_table_candidate.json').read_text())
    original={name:np.r_[d.xpos[m.body(name).id],d.xquat[m.body(name).id]] for name in meshes}
    base={'white_pot':np.array(initial['candidate_pot_pose']),'white_pot_lid':original['white_pot_lid'].copy()}
    observations={};surface={};columns={};ids={}
    for k,name in enumerate(meshes):
        points=np.load(OUT/f'{name}_depth_points.npz')['points_source']+shift
        height=points@normal+setup_offset
        points=points[(height>.015)&(height<.16)][::max(1,len(points)//1200)]
        observations[name]=points
        sample,faces=trimesh.sample.sample_surface(meshes[name],180000,seed=31+k)
        surface[name]=(sample,meshes[name].face_normals[faces],cKDTree(sample))
        columns[name]=int(m.joint(name+'_free').qposadr[0]);ids[name]=m.body(name).id
    pairids={(int(g1),int(g2)):i for i,(g1,g2) in enumerate((g1,g2) for g1 in np.flatnonzero(m.geom_bodyid==ids['white_pot']) for g2 in np.flatnonzero(m.geom_bodyid==ids['white_pot_lid']))}
    source=cv2.imread(str(OUT/'source_000.jpg'))
    target_masks={}
    for name,path in cfg['object_masks'].items():
        mm=cv2.imread(str(Path(path)/'000000.png'),0)
        target_masks[name]=cv2.resize(mm,(source.shape[1],source.shape[0]),interpolation=cv2.INTER_NEAREST)>0

    def poses(x):
        result={}
        for i,name in enumerate(meshes):
            q=base[name].copy();q[:3]+=x[i*4:i*4+3]
            q[3:]=(Rotation.from_rotvec(normal*x[i*4+3])*Rotation.from_quat(q[3:],scalar_first=True)).as_quat(scalar_first=True)
            result[name]=q
        return result

    def residual(x,report=False):
        ps=poses(x);depth=[];details={};pieces=[]
        for name,q in ps.items():
            R=Rotation.from_quat(q[3:],scalar_first=True).as_matrix();local=(observations[name]-q[:3])@R
            sampled,normals,tree=surface[name];dist,index=tree.query(local)
            signed=np.sum((local-sampled[index])*normals[index],axis=1)
            # Signed surface distance plus a small tangential term to constrain
            # the visible handle/rim without penalizing occluded back surfaces.
            depth.extend([np.clip(signed,-.012,.012),.15*np.minimum(dist,.02)])
            d.qpos[columns[name]:columns[name]+7]=q
            details[name]={'median_stereo_surface_distance_m':float(np.median(dist)),
                'p90_stereo_surface_distance_m':float(np.quantile(dist,.9))}
        mujoco.mj_forward(m,d)
        collision=np.zeros(len(pairids))
        for c in d.contact:
            i=pairids.get((int(c.geom1),int(c.geom2)))
            if i is not None:collision[i]=max(collision[i],max(0.,-float(c.dist)-.0001))
        q=ps['white_pot'];R=Rotation.from_quat(q[3:],scalar_first=True).as_matrix()
        bottom=float(np.min((meshes['white_pot'].vertices@R.T+q[:3])@normal+setup_offset))
        # This is registration, not a force/controller optimization: the only
        # feasibility terms are rigid nonintersection and measured table support.
        pieces.extend(depth);pieces.extend([collision*40,np.array([bottom*30]),x*.05])
        if report:
            details['max_pot_lid_penetration_m']=float(collision.max()+.0001 if collision.max()>0 else 0.)
            details['pot_bottom_vs_stereo_plane_m']=bottom
            projected={name:mask(meshes[name],q,camera,source.shape)>0 for name,q in ps.items()}
            predicted_visible={'white_pot':projected['white_pot']&~projected['white_pot_lid'],'white_pot_lid':projected['white_pot_lid']}
            for name,pred in predicted_visible.items():
                truth=target_masks[name]
                details[name]['visible_mask_iou']=float((pred&truth).sum()/max(1,(pred|truth).sum()))
            return details
        return np.concatenate([np.asarray(p).ravel() for p in pieces])

    x0=np.zeros(8);x0[6]=.01
    before=residual(np.zeros(8),True)
    fit=least_squares(residual,x0,bounds=(np.tile([-.025,-.025,-.02,-.12],2),np.tile([.025,.025,.035,.12],2)),
                      x_scale=np.tile([.01,.01,.01,.05],2),diff_step=.002,max_nfev=180,verbose=1,ftol=1e-7,xtol=1e-8,gtol=1e-7)
    result=poses(fit.x);after=residual(fit.x,True)
    panels=[]
    for title,ps in [('Original input',original),('Plane-corrected pot only',base),('Stereo + support + collision registration',result)]:
        panel=source.copy()
        for name,color in [('white_pot',(255,150,0)),('white_pot_lid',(0,0,255))]:outline(panel,meshes[name],ps[name],camera,color)
        cv2.putText(panel,title,(650,740),cv2.FONT_HERSHEY_SIMPLEX,.63,(0,255,255),2)
        panels.append(cv2.resize(panel[700:1400,650:1450],(640,560)))
    cv2.imwrite(str(OUT/'fitted_seated_registration_overlay.jpg'),np.hstack(panels))
    output=dict(provisional=True,method='fixed source mesh stereo point-to-surface registration + measured table support + no pot/lid penetration',
                initial_poses={k:v.tolist() for k,v in original.items()},candidate_poses={k:v.tolist() for k,v in result.items()},
                normal_source=normal.tolist(),plane_offset_source=offset,normal_setup=normal.tolist(),plane_offset_setup=setup_offset,
                plane_aligned_initial_stats=before,fitted_stats=after,parameters=fit.x.tolist(),optimizer_success=bool(fit.success),
                assets_changed=False,source_files_changed=False,trajectory_targets_changed=False)
    (OUT/'fitted_seated_registration.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':run()
