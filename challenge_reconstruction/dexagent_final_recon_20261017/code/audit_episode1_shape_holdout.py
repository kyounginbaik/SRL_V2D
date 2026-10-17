"""Test a frame0-fitted shape on untouched later RGB/stereo observations."""
from pathlib import Path
import argparse,json
import cv2,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from repair_episode1_registration import B,EP,OUT,mask,outline

def transform(p):
    result=np.eye(4);result[:3,:3]=Rotation.from_quat(p[3:],scalar_first=True).as_matrix();result[:3,3]=p[:3];return result

def audit(version):
    folder=OUT/version;fit=json.loads((folder/'fit.json').read_text());source=B.parent/'final_recon/episode_000001'
    cfg=json.loads((source/'config.json').read_text());poses=np.load(source/'object_poses.npz');names=list(poses['names'].astype(str));pi=names.index('white_pot');li=names.index('white_pot_lid')
    original=trimesh.load(source/'assets/white_pot_visual.obj',force='mesh',process=False);repaired=trimesh.load(folder/'white_pot_visual.obj',force='mesh',process=False);lid=trimesh.load(source/'assets/white_pot_lid_visual.obj',force='mesh',process=False)
    shift=np.array(json.loads((EP/'input_manifest.json').read_text())['setup_from_source'])[:3,3];candidate=transform(fit['pose']);candidate[:3,3]-=shift
    correction=np.linalg.inv(poses['world_from_object'][0,pi])@candidate
    rng=np.random.default_rng(91);trees=[]
    for mesh in [original,repaired]:
        points,_=trimesh.sample.sample_surface(mesh,150000,seed=81);trees.append(cKDTree(points))
    reports=[];rows=[];clouds=OUT/'multiframe_clouds'
    for frame in [0,60,100,120,160,200,260]:
        archive=clouds/f'white_pot_{frame:06d}.npz'
        if not archive.exists():continue
        z=np.load(archive);points=z['points_source'];p=poses['world_from_object'][frame,pi];new=p@correction
        observed=points[rng.choice(len(points),min(2000,len(points)),replace=False)]
        camera={'world_from_rgb':z['world_from_rgb'][None],'K':z['K']}
        rgb=cv2.imread(str(clouds/f'rgb_{frame:06d}.jpg'));h,w=rgb.shape[:2]
        truth=cv2.resize(cv2.imread(str(Path(cfg['object_masks']['white_pot'])/f'{frame:06d}.png'),0),(w,h),interpolation=cv2.INTER_NEAREST)>0
        ignore=np.zeros((h,w),np.uint8)
        for track in cfg['hand_tracks']:
            path=Path(cfg['source'])/'masks'/track/f'{frame:06d}.png';hm=cv2.imread(str(path),0)
            if hm is not None:ignore|=cv2.resize(hm,(w,h),interpolation=cv2.INTER_NEAREST)
        valid=cv2.dilate(ignore,np.ones((9,9),np.uint8))==0
        lp=poses['world_from_object'][frame,li];lp=np.r_[lp[:3,3],Rotation.from_matrix(lp[:3,:3]).as_quat(scalar_first=True)]
        lm=mask(lid,lp,camera,rgb.shape)>0
        record={'frame':frame,'time_s':frame/20.};panels=[]
        for label,mesh,T,tree in [('original',original,p,trees[0]),('repaired',repaired,new,trees[1])]:
            distance=tree.query((observed-T[:3,3])@T[:3,:3])[0]
            pose=np.r_[T[:3,3],Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True)]
            pred=(mask(mesh,pose,camera,rgb.shape)>0)&~lm&valid;target=truth&valid
            record[label]=dict(median_surface_distance_m=float(np.median(distance)),p90_surface_distance_m=float(np.quantile(distance,.9)),visible_mask_iou=float((pred&target).sum()/max(1,(pred|target).sum())))
            picture=rgb.copy();outline(picture,mesh,pose,camera,(255,180,0) if label=='original' else (0,255,0));cv2.putText(picture,f'{label} {frame/20:.1f}s',(550,670),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)
            panels.append(cv2.resize(picture[650:1500,500:1700],(600,425)))
        rows.append(np.hstack(panels));reports.append(record)
    cv2.imwrite(str(folder/'holdout_overlay.jpg'),np.vstack(rows))
    (folder/'holdout_audit.json').write_text(json.dumps(dict(method='fixed local pose correction learned at frame0, applied unchanged to original poses; original masks/stereo held out',records=reports),indent=2))
    print(json.dumps(reports,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',default='pot_geometry_rgb_depth_v3');audit(p.parse_args().version)
