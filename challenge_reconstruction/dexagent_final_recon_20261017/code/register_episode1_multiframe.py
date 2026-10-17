"""Re-register fixed shape candidates to additional observed stereo surfaces."""
from pathlib import Path
import argparse,json
import cv2,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,EP,OUT,outline,mask

def rigid_fit(source,target):
    a=source.mean(0);b=target.mean(0);u,s,vt=np.linalg.svd((source-a).T@(target-b));R=vt.T@u.T
    if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
    return R,b-R@a

def run(version):
    folder=OUT/version;fit=json.loads((folder/'fit.json').read_text());src=B.parent/'final_recon/episode_000001';cfg=json.loads((src/'config.json').read_text())
    z=np.load(src/'object_poses.npz');names=list(z['names'].astype(str));oi=names.index('white_pot');li=names.index('white_pot_lid')
    shift=np.array(json.loads((EP/'input_manifest.json').read_text())['setup_from_source'])[:3,3]
    first=np.eye(4);first[:3,:3]=Rotation.from_quat(fit['pose'][3:],scalar_first=True).as_matrix();first[:3,3]=np.array(fit['pose'][:3])-shift
    correction=np.linalg.inv(z['world_from_object'][0,oi])@first
    paths={'original':src/'assets/white_pot_visual.obj','candidate':folder/'white_pot_visual.obj'}
    meshes={k:trimesh.load(v,force='mesh',process=False) for k,v in paths.items()};sampled={k:trimesh.sample.sample_surface(mesh,200000,seed=71)[0] for k,mesh in meshes.items()};trees={k:cKDTree(v) for k,v in sampled.items()}
    lid=trimesh.load(src/'assets/white_pot_lid_visual.obj',force='mesh',process=False)
    reports=[];rows=[]
    for frame in [60,100,120,160,200,260]:
        archive=OUT/f'multiframe_clouds/white_pot_{frame:06d}.npz';obs=np.load(archive);points=obs['points_source'][::max(1,len(obs['points_source'])//2500)]
        camera={'world_from_rgb':obs['world_from_rgb'][None],'K':obs['K']};rgb=cv2.imread(str(OUT/f'multiframe_clouds/rgb_{frame:06d}.jpg'));h,w=rgb.shape[:2]
        target=cv2.resize(cv2.imread(str(Path(cfg['object_masks']['white_pot'])/f'{frame:06d}.png'),0),(w,h),interpolation=cv2.INTER_NEAREST)>0
        valid=np.ones((h,w),bool)
        for track in cfg['hand_tracks']:
            mm=cv2.imread(str(Path(cfg['source'])/'masks'/track/f'{frame:06d}.png'),0)
            if mm is not None:valid&=cv2.dilate(cv2.resize(mm,(w,h),interpolation=cv2.INTER_NEAREST),np.ones((9,9),np.uint8))==0
        lp=z['world_from_object'][frame,li];lp=np.r_[lp[:3,3],Rotation.from_matrix(lp[:3,:3]).as_quat(scalar_first=True)];lm=mask(lid,lp,camera,rgb.shape)>0
        report={'frame':frame};panels=[]
        for label in paths:
            T=z['world_from_object'][frame,oi].copy()
            if label=='candidate':T=T@correction
            starting=T.copy()
            for iteration in range(35):
                local=(points-T[:3,3])@T[:3,:3];dist,index=trees[label].query(local)
                keep=dist<min(.025,np.quantile(dist,.85));a=sampled[label][index[keep]];b=points[keep]
                R,p=rigid_fit(a,b);change=np.linalg.norm(p-T[:3,3])+np.linalg.norm(R-T[:3,:3]);T[:3,:3]=R;T[:3,3]=p
                if change<1e-6:break
            dist=trees[label].query((points-T[:3,3])@T[:3,:3])[0];pose=np.r_[T[:3,3],Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True)]
            pred=(mask(meshes[label],pose,camera,rgb.shape)>0)&~lm&valid;truth=target&valid
            report[label]=dict(median_stereo_distance_m=float(np.median(dist)),p80_stereo_distance_m=float(np.quantile(dist,.8)),mask_iou=float((pred&truth).sum()/max(1,(pred|truth).sum())),world_from_object=T.tolist(),
                registration_translation_change_m=float(np.linalg.norm(T[:3,3]-starting[:3,3])),registration_rotation_change_deg=float(np.rad2deg(Rotation.from_matrix(T[:3,:3]@starting[:3,:3].T).magnitude())))
            panel=rgb.copy();outline(panel,meshes[label],pose,camera,(255,180,0) if label=='original' else (0,255,0));cv2.putText(panel,f'{label} ICP {frame/20:.1f}s',(550,670),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2);panels.append(cv2.resize(panel[650:1500,500:1700],(600,425)))
        rows.append(np.hstack(panels));reports.append(report);print(frame,{k:(report[k]['median_stereo_distance_m'],report[k]['mask_iou']) for k in paths},flush=True)
    cv2.imwrite(str(folder/'multiframe_registration_overlay.jpg'),np.vstack(rows));(folder/'multiframe_registration.json').write_text(json.dumps(dict(method='Same fixed mesh, per-frame robust point-to-point ICP initialized from reference; no physics objective',records=reports),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',default='pot_geometry_rgb_depth_v3');run(p.parse_args().version)
