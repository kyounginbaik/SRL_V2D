"""Versioned multi-frame RGB-D shape/registration audit; no physics objective.

Alternating robust correspondences fit one fixed pot shape across seven frames.
Frame zero alone has the independently measured support-plane constraint.
Other frames have independent rigid poses because the human moves the pot.
Nothing is written into final_recon.
"""
from pathlib import Path
import argparse, json
import cv2, numpy as np, trimesh
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B, EP, OUT, mask, outline


def run(iterations=12):
    src=B.parent/'final_recon/episode_000001'; dest=OUT/'multiframe_shape_v1';dest.mkdir(exist_ok=True)
    cfg=json.loads((src/'config.json').read_text()); mesh=trimesh.load(src/'assets/white_pot_visual.obj',force='mesh',process=False)
    fit=json.loads((OUT/'pot_geometry_rgb_depth_v3/fit.json').read_text()); axis=np.array(fit['local_height_axis']);floor=fit['local_floor']
    u=np.array(fit['local_tangent_u']);v=np.array(fit['local_tangent_v']);center=np.array(fit['local_bowl_center'])
    support=json.loads((B/'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json').read_text())
    table=np.array(support['proposed_setup_from_source']);n=table[2,:3];tangent=table[:2,:3]
    shift=np.array(json.loads((EP/'input_manifest.json').read_text())['setup_from_source'])[:3,3]
    base=np.array(json.loads((OUT/'pot_table_candidate.json').read_text())['candidate_pot_pose']);base[:3]-=shift
    ref=np.load(src/'object_poses.npz');oi=list(ref['names'].astype(str)).index('white_pot')
    frames=[0,60,100,120,160,200,260];clouds=[];samples,_=trimesh.sample.sample_surface(mesh,120000,seed=610)
    for f in frames:
        obs=np.load(OUT/f'multiframe_clouds/white_pot_{f:06d}.npz');p=obs['points_source']
        # Spatially deterministic subsampling, equal weight per frame.
        clouds.append(p[np.linspace(0,len(p)-1,min(1800,len(p))).astype(int)])
    # shape6 = height, handle elevation/slope, two handle tangent shifts, radius
    shape0=np.array(fit['parameters'])[3:]
    x=np.r_[np.array(fit['parameters'])[:3],shape0,np.zeros(6*(len(frames)-1))]
    lower=np.r_[[-.025,-.025,-.2],[.7,-.01,-.35,-.025,-.025,.9],np.tile([-.04]*3+[-.45]*3,len(frames)-1)]
    upper=np.r_[[.025,.025,.2],[1.15,.055,.5,.025,.025,1.15],np.tile([.04]*3+[.45]*3,len(frames)-1)]
    prior_poses=[]
    for f in frames:
        P=ref['world_from_object'][f,oi].copy()
        prior_poses.append(P)

    def warp(verts,s):
        height=verts@axis-floor;w=np.clip((verts[:,2]-.075)/.055,0,1);w=w*w*(3-2*w)
        radial=verts-center;radial-=(radial@axis)[:,None]*axis
        return verts+((s[0]-1)*height*(1-w)+w*(s[1]+s[2]*(verts[:,2]-.13)))[:,None]*axis+w[:,None]*(s[3]*u+s[4]*v)+(1-w[:,None])*(s[5]-1)*radial

    def poses(x):
        ans=[];R=Rotation.from_rotvec(n*x[2])*Rotation.from_quat(base[3:],scalar_first=True)
        p=base[:3]+x[:2]@tangent;verts=warp(mesh.vertices,x[3:9]);bottom=np.min((verts@R.as_matrix().T+p)@n+table[2,3]);p+=(.0001-bottom)*n
        ans.append((R.as_matrix(),p))
        for k in range(1,len(frames)):
            delta=x[9+6*(k-1):9+6*k];P=prior_poses[k]
            R=(Rotation.from_rotvec(delta[3:])*Rotation.from_matrix(P[:3,:3])).as_matrix();ans.append((R,P[:3,3]+delta[:3]))
        return ans

    history=[]
    for it in range(iterations):
        current=warp(samples,x[3:9]);tree=cKDTree(current);matches=[]
        for k,(R,p) in enumerate(poses(x)):
            local=(clouds[k]-p)@R;dist,index=tree.query(local)
            keep=dist<min(.025,np.quantile(dist,.88))
            matches.append((samples[index[keep]],clouds[k][keep]))
        def residual(xx):
            errors=[]
            for (a,b),(R,p) in zip(matches,poses(xx)):
                errors.append(((warp(a,xx[3:9])@R.T+p-b)/.003).ravel()/np.sqrt(len(a)))
            # Modest evidence-independent prior against unnecessary deformation.
            s=xx[3:9];errors.append(.1*(s-np.array([1.,0.,0.,0.,0.,1.]))/np.array([.15,.03,.3,.02,.02,.1]))
            return np.concatenate(errors)
        result=least_squares(residual,x,bounds=(lower,upper),max_nfev=24,loss='soft_l1',f_scale=.2,diff_step=1e-4)
        x=result.x;stats=[];tree=cKDTree(warp(samples,x[3:9]))
        for f,pc,(R,p) in zip(frames,clouds,poses(x)):
            dist=tree.query((pc-p)@R)[0];stats.append(dict(frame=f,median_m=float(np.median(dist)),p80_m=float(np.quantile(dist,.8))))
        row=dict(iteration=it,parameters=x.tolist(),cost=float(result.cost),frames=stats);history.append(row)
        (dest/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    candidate=mesh.copy();candidate.vertices=warp(mesh.vertices,x[3:9]);candidate.export(dest/'white_pot_visual.obj')
    rows=[];records=[]
    for f,pc,(R,p) in zip(frames,clouds,poses(x)):
        obs=np.load(OUT/f'multiframe_clouds/white_pot_{f:06d}.npz');camera={'K':obs['K'],'world_from_rgb':obs['world_from_rgb'][None]}
        rgb=cv2.imread(str(OUT/f'multiframe_clouds/rgb_{f:06d}.jpg'));pose=np.r_[p,Rotation.from_matrix(R).as_quat(scalar_first=True)]
        panel=rgb.copy();outline(panel,candidate,pose,camera,(0,255,0));cv2.putText(panel,f'Multiframe shape, {f/20:g}s',(550,680),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
        rows.append(cv2.resize(panel[650:1500,500:1700],(600,425)))
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=p;records.append(dict(frame=f,world_from_object=T.tolist()))
    cv2.imwrite(str(dest/'rgb_overlay.jpg'),np.vstack(rows));fit.update(parameters=x[:9].tolist(),pose=np.r_[poses(x)[0][1]+shift,Rotation.from_matrix(poses(x)[0][0]).as_quat(scalar_first=True)].tolist(),
        ranking_data='Seven RGB-D frames with one shared shape; frame0 independently measured tabletop support. No physical/task reward.',provisional=True,
        method='Alternating robust stereo correspondence and bounded least squares; masks rendered for audit only',frames=records,statistics=history[-1]['frames'])
    (dest/'fit.json').write_text(json.dumps(fit,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iterations',type=int,default=12);run(**vars(p.parse_args()))
