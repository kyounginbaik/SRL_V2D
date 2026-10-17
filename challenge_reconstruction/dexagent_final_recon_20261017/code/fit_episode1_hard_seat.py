"""Hard-support/nonintersection counterpart to the diagnostic soft pose fit."""
from pathlib import Path
import json
import cv2
import mujoco
import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,EP,OUT,load,outline,mask


def run():
    m,d,camera,meshes=load();source=cv2.imread(str(OUT/'source_000.jpg'))
    support=json.loads((B/'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json').read_text())
    n=np.array(support['fitted_plane_normal_source']);T=np.array(support['proposed_setup_from_source'])
    shift=np.array(json.loads((EP/'input_manifest.json').read_text())['setup_from_source'])[:3,3]
    offset=T[2,3]-n@shift;tangent=T[:2,:3]
    original={o:np.r_[d.xpos[m.body(o).id],d.xquat[m.body(o).id]] for o in meshes}
    base={**original,'white_pot':np.array(json.loads((OUT/'pot_table_candidate.json').read_text())['candidate_pot_pose'])}
    obs={};trees={};collision_vertices={};cols={};bids={}
    for i,o in enumerate(meshes):
        pc=np.load(OUT/f'{o}_depth_points.npz')['points_source']+shift;h=pc@n+offset;pc=pc[(h>.015)&(h<.16)]
        obs[o]=pc[::max(1,len(pc)//1000)]
        pts,_=trimesh.sample.sample_surface(meshes[o],160000,seed=71+i);trees[o]=cKDTree(pts)
        collision_vertices[o]=np.vstack([trimesh.load(p,force='mesh',process=False).vertices for p in sorted((B.parent/f'final_recon/episode_000001/assets/{o}_convex').glob('*.obj'))])
        cols[o]=int(m.joint(o+'_free').qposadr[0]);bids[o]=m.body(o).id
    cfg=json.loads((B.parent/'final_recon/episode_000001/config.json').read_text())
    small_camera={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in camera.items()};small_camera['K']=camera['K'].copy();small_camera['K'][:2]/=3
    small_shape=(source.shape[0]//3,source.shape[1]//3,3)
    truth={o:cv2.resize(cv2.imread(str(Path(p)/'000000.png'),0),(small_shape[1],small_shape[0]),interpolation=cv2.INTER_NEAREST)>0 for o,p in cfg['object_masks'].items()}
    target_bodies=set(bids.values())

    def make_pose(x):
        ps={}
        for i,o in enumerate(meshes):
            q=base[o].copy();q[:3]+=x[i*3:i*3+2]@tangent
            q[3:]=(Rotation.from_rotvec(n*x[i*3+2])*Rotation.from_quat(q[3:],scalar_first=True)).as_quat(scalar_first=True)
            ps[o]=q
        o='white_pot';q=ps[o];R=Rotation.from_quat(q[3:],scalar_first=True).as_matrix()
        q[:3]+=(.0001-np.min((collision_vertices[o]@R.T+q[:3])@n+offset))*n
        d.qpos[cols[o]:cols[o]+7]=q
        lidbase=ps['white_pot_lid'].copy()
        def penetration(z):
            q=lidbase.copy();q[:3]+=z*n;d.qpos[cols['white_pot_lid']:cols['white_pot_lid']+7]=q
            mujoco.mj_forward(m,d)
            return max([max(0.,-float(c.dist)) for c in d.contact if {int(m.geom_bodyid[c.geom1]),int(m.geom_bodyid[c.geom2])}==target_bodies]+[0.])
        lo=-.01;hi=.06
        if penetration(hi)>.0001:raise RuntimeError('Cannot find separated seat above pot')
        for _ in range(13):
            mid=(lo+hi)/2
            if penetration(mid)>.0001:lo=mid
            else:hi=mid
        ps['white_pot_lid'][:3]+=(hi+.0002)*n
        return ps

    def stats(ps):
        result={}
        projected={o:mask(meshes[o],q,small_camera,small_shape)>0 for o,q in ps.items()}
        visible={'white_pot':projected['white_pot']&~projected['white_pot_lid'],'white_pot_lid':projected['white_pot_lid']}
        for o,q in ps.items():
            R=Rotation.from_quat(q[3:],scalar_first=True).as_matrix();distance=trees[o].query((obs[o]-q[:3])@R)[0]
            result[o]=dict(median_stereo_surface_distance_m=float(np.median(distance)),clipped_mean_stereo_surface_distance_m=float(np.minimum(distance,.025).mean()),
                           visible_mask_iou=float((visible[o]&truth[o]).sum()/max(1,(visible[o]|truth[o]).sum())))
        return result
    history=[]
    def objective(x):
        ps=make_pose(x);ss=stats(ps)
        # The support/nonintersection are hard construction constraints. Only
        # observed image/depth fit ranks physically feasible registrations.
        cost=sum(600*s['clipped_mean_stereo_surface_distance_m']+12*(1-s['visible_mask_iou']) for s in ss.values())
        history.append(dict(cost=cost,parameters=x.tolist()))
        if len(history)%25==0:print('evaluation',len(history),'cost',cost,'best',min(h['cost'] for h in history),flush=True)
        return cost
    x0=np.zeros(6)
    fit=minimize(objective,x0,method='Powell',bounds=[(-.025,.025),(-.025,.025),(-.15,.15)]*2,
                 options={'maxiter':8,'maxfev':800,'xtol':2e-4,'ftol':1e-4})
    best=min(history,key=lambda r:r['cost']);ps=make_pose(np.array(best['parameters']));after=stats(ps);before=stats(original)
    panels=[]
    for title,p in [('Original input',original),('Hard-supported seated registration',ps)]:
        panel=source.copy()
        for o,c in [('white_pot',(255,150,0)),('white_pot_lid',(0,0,255))]:outline(panel,meshes[o],p[o],camera,c)
        cv2.putText(panel,title,(650,740),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
        panels.append(cv2.resize(panel[700:1400,650:1450],(800,700)))
    cv2.imwrite(str(OUT/'hard_seated_registration_overlay.jpg'),np.hstack(panels))
    result=dict(provisional=True,method='Fixed visual/convex assets; exact collision support on measured stereo plane; seated lid first nonintersecting contact along plane normal; RGB mask + stereo surface fit in tangent translation/yaw',
        original_poses={k:v.tolist() for k,v in original.items()},candidate_poses={k:v.tolist() for k,v in ps.items()},
        original_stats=before,candidate_stats=after,normal_setup=n.tolist(),plane_offset_setup=offset,
        parameters=best['parameters'],cost=best['cost'],evaluations=len(history),optimizer_success=bool(fit.success),
        assets_changed=False,source_changed=False,object_actuation=False)
    (OUT/'hard_seated_registration.json').write_text(json.dumps(result,indent=2));(OUT/'hard_fit_history.json').write_text(json.dumps(history))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':run()
