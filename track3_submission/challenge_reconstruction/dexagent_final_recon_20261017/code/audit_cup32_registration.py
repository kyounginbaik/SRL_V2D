#!/usr/bin/env python3
"""Read-only stereo/mask comparison of cup32's tabletop penetration."""
from pathlib import Path
import json,cv2,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from repair_episode1_registration import mask,outline
B=Path(__file__).resolve().parents[1];SOURCE=B.parent/'final_recon/episode_000032'


def main():
    out=B/'refinement_20261018/expanded_batch/source_review/episode_000032/registration_audit';out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((SOURCE/'config.json').read_text());manifest=json.loads((B/'episodes/episode_000032/input_manifest.json').read_text())
    camera=dict(np.load(B/'episodes/episode_000032/camera_trajectory.npz'));transform=np.asarray(manifest['setup_from_source'])
    obj=np.load(SOURCE/'object_poses.npz');idx=obj['names'].tolist().index('blue_cup');T=transform@obj['world_from_object'][0,idx]
    mesh=trimesh.load(SOURCE/'assets/blue_cup.obj',process=False);pose=np.r_[T[:3,3],Rotation.from_matrix(T[:3,:3]).as_quat(scalar_first=True)]
    cap=cv2.VideoCapture(str(Path(cfg['source'])/'anycalib/undistorted.mp4'));ok,rgb=cap.read();cap.release()
    if not ok:raise ValueError('Missing original RGB')
    h,w=rgb.shape[:2];observed=cv2.imread(str(Path(cfg['object_masks']['blue_cup'])/'000000.png'),0);observed=cv2.resize(observed,(w,h),interpolation=cv2.INTER_NEAREST)>0
    hands=np.zeros((h,w),np.uint8)
    for side in cfg['hand_tracks']:
        p=Path(cfg['source'])/'masks'/side/'000000.png';im=cv2.imread(str(p),0)
        if im is not None:hands|=cv2.resize(im,(w,h),interpolation=cv2.INTER_NEAREST)
    visible=observed&~(cv2.dilate(hands,np.ones((5,5),np.uint8))>0)
    raw=cv2.imread(str(Path(cfg['depth'])/'000000.png'),-1)
    with np.errstate(divide='ignore'):depth=65535/raw.astype(float)-1
    depth=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
    ys,xs=np.where(visible&(depth>.1)&(depth<2));stride=max(1,len(xs)//5000);ys,xs=ys[::stride],xs[::stride]
    z=depth[ys,xs];K=camera['K'];cam=np.c_[(xs-K[0,2])*z/K[0,0],(ys-K[1,2])*z/K[1,1],z]
    C=camera['world_from_rgb'][0];points=cam@C[:3,:3].T+C[:3,3]
    samples,_=trimesh.sample.sample_surface(mesh,100000,seed=43);surface=samples@T[:3,:3].T+T[:3,3]
    rows=[];panels=[]
    for dz in np.arange(0,.0401,.002):
        candidate=pose.copy();candidate[2]+=dz;projected=(mask(mesh,candidate,camera,rgb.shape)>0)&~(hands>0)
        distance=cKDTree(surface+np.array([0,0,dz])).query(points)[0]
        intersection=(projected&visible).sum();union=(projected|visible).sum()
        rows.append(dict(translation_setup_z_m=float(dz),visible_mask_iou=float(intersection/max(1,union)),
            stereo_surface_distance_median_m=float(np.median(distance)),stereo_surface_distance_p80_m=float(np.quantile(distance,.8)),
            mesh_bottom_m=float((mesh.vertices@T[:3,:3].T+candidate[:3])[:,2].min())))
    for dz in [0.,.018,.026]:
        candidate=pose.copy();candidate[2]+=dz;image=rgb.copy();outline(image,mesh,candidate,camera,(0,0,255))
        ys2,xs2=np.where(observed);x0=max(0,int(xs2.min())-80);x1=min(w,int(xs2.max())+80);y0=max(0,int(ys2.min())-80);y1=min(h,int(ys2.max())+80)
        panel=cv2.resize(image[y0:y1,x0:x1],(480,480));cv2.putText(panel,f'Cup-only z +{dz*1000:.0f} mm',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2);panels.append(panel)
    cv2.imwrite(str(out/'registration_candidates.jpg'),np.hstack(panels))
    report=dict(episode='episode_000032',source_unchanged=True,planning_only=True,depth_points=len(points),scan=rows,
        stereo_best=min(rows,key=lambda r:r['stereo_surface_distance_median_m']),mask_best=max(rows,key=lambda r:r['visible_mask_iou']),
        scope='One initial visible-frame translation diagnostic, not an adopted reconstruction repair or metric target change')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='scan'},indent=2))


if __name__=='__main__':main()
