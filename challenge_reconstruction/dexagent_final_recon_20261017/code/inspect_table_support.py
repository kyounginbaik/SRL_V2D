#!/usr/bin/env python3
"""Read-only tabletop support estimate from calibrated RGB stereo depth.

Object and hand masks are excluded before fitting. This estimates a simulator
support surface; it never changes reconstructed object targets or source files.
"""
from pathlib import Path
import argparse
import json
import hashlib
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import trimesh

B=Path(__file__).resolve().parents[1]
ROOT=B.parents[1]


def inspect(episode):
    name=f'episode_{episode:06d}';source=ROOT/'challenge_reconstruction/final_recon'/name
    out=B/'refinement_20261018/expanded_batch/source_review'/name;out.mkdir(parents=True,exist_ok=True)
    if (out/'table_support.json').exists():
        existing=json.loads((out/'table_support.json').read_text())
        if existing['source_pose_sha256']!=hashlib.sha256((source/'object_poses.npz').read_bytes()).hexdigest():
            raise ValueError('Source poses changed; use a new versioned support-review directory')
        print(json.dumps(existing),flush=True)
        return existing
    cfg=json.loads((source/'config.json').read_text());cal=json.loads((source/'stereo_calibration/rgb_intrinsics.json').read_text())
    camera=np.load(source/'camera_motion.npz')['world_from_rgb'];poses=np.load(source/'object_poses.npz')
    capture=cv2.VideoCapture(str(Path(cfg['source'])/'anycalib/undistorted.mp4'))
    frame_reports=[];all_points=[]
    for frame in [0,5,10]:
        capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,rgb=capture.read()
        if not ok:raise ValueError('Source RGB frame missing')
        h,w=rgb.shape[:2];K=np.array([[cal['fx'],0,cal['cx']],[0,cal['fy'],cal['cy']],[0,0,1.]])
        K[0]*=w/cal['width'];K[1]*=h/cal['height']
        hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV);mask=cv2.inRange(hsv,np.array([4,65,35],np.uint8),np.array([35,255,255],np.uint8))
        mask[:int(.35*h)]=0;mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        selected=np.zeros_like(mask);cv2.drawContours(selected,[max(contours,key=cv2.contourArea)],-1,255,-1)
        selected=cv2.erode(selected,np.ones((15,15),np.uint8))>0
        for folder in [*cfg['object_masks'].values(),*[str(Path(cfg['source'])/'masks'/x) for x in cfg['hand_tracks']]]:
            mm=cv2.imread(str(Path(folder)/f'{frame:06d}.png'),0)
            if mm is None:continue
            mm=cv2.resize(mm,(w,h),interpolation=cv2.INTER_NEAREST)
            selected &= ~(cv2.dilate(mm,np.ones((21,21),np.uint8))>0)
        raw=cv2.imread(str(Path(cfg['depth'])/f'{frame:06d}.png'),-1)
        with np.errstate(divide='ignore',invalid='ignore'):depth=65535/raw.astype(float)-1
        depth=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
        ys,xs=np.where(selected&np.isfinite(depth)&(depth>.1)&(depth<2))
        stride=max(1,len(xs)//15000);xs,ys=xs[::stride],ys[::stride];d=depth[ys,xs]
        points=np.c_[(xs-K[0,2])*d/K[0,0],(ys-K[1,2])*d/K[1,1],d]
        world=points@camera[frame,:3,:3].T+camera[frame,:3,3]
        # Reject isolated depth outliers before robust plane fitting.
        center=np.median(world[:,2]);world=world[abs(world[:,2]-center)<.1]
        all_points.append(world)
        xx=np.c_[world[:,:2],np.ones(len(world))]
        local_fit=least_squares(lambda p:xx@p-world[:,2],[0,0,np.median(world[:,2])],loss='soft_l1',f_scale=.005)
        local_normal=np.r_[-local_fit.x[:2],1.];local_normal/=np.linalg.norm(local_normal)
        overlay=rgb.copy();overlay[selected]=(.45*overlay[selected]+.55*np.array([30,220,30])).astype(np.uint8)
        cv2.imwrite(str(out/f'table_mask_{frame:06d}.jpg'),overlay)
        frame_reports.append(dict(frame=frame,points=len(world),median_z_m=float(np.median(world[:,2])),
            plane_z_coefficients=local_fit.x.tolist(),normal=local_normal.tolist(),
            plane_median_residual_m=float(np.median(abs(xx@local_fit.x-world[:,2])))))
    capture.release();points=np.vstack(all_points);X=np.c_[points[:,:2],np.ones(len(points))]
    fit=least_squares(lambda p:X@p-points[:,2],[0,0,np.median(points[:,2])],loss='soft_l1',f_scale=.005)
    residual=np.abs(X@fit.x-points[:,2]);inlier=residual<.015
    horizontal_z=float(np.median(points[inlier,2]));horizontal_error=abs(points[inlier,2]-horizontal_z)
    tilt=float(np.degrees(np.arctan(np.linalg.norm(fit.x[:2]))))
    normal=np.r_[-fit.x[:2],1.];normal/=np.linalg.norm(normal)
    rot,_=Rotation.align_vectors([[0,0,1]],[normal])
    transform=np.eye(4);transform[:3,:3]=rot.as_matrix();transform[2,3]=-np.median(points[inlier]@normal)
    normal_spread=max(float(np.degrees(np.arccos(np.clip(normal@np.asarray(r['normal']),-1,1)))) for r in frame_reports)
    bottoms={}
    aligned_bottoms={}
    for i,obj in enumerate(poses['names']):
        mesh=trimesh.load(source/'assets'/f'{obj}.obj',process=False);T=poses['world_from_object'][0,i]
        bottoms[str(obj)]=float((mesh.vertices@T[:3,:3].T+T[:3,3])[:,2].min())
        transformed=transform@T
        aligned_bottoms[str(obj)]=float((mesh.vertices@transformed[:3,:3].T+transformed[:3,3])[:,2].min())
    report=dict(episode=name,method='RGB largest wooden-table component, exclude dilated object/hand masks, calibrated stereo backprojection, robust plane; horizontal z median inliers',
        source_pose_sha256=hashlib.sha256((source/'object_poses.npz').read_bytes()).hexdigest(),
        frames=frame_reports,points=len(points),plane_z_coefficients=fit.x.tolist(),plane_tilt_degrees=tilt,
        plane_median_absolute_residual_m=float(np.median(residual)),inlier_fraction_15mm=float(inlier.mean()),
        horizontal_support_z_m=horizontal_z,horizontal_residual_p50_p95_m=np.quantile(horizontal_error,[.5,.95]).tolist(),
        initial_mesh_bottoms_m=bottoms,initial_bottom_minus_stereo_table_m={k:v-horizontal_z for k,v in bottoms.items()},
        fitted_plane_normal_source=normal.tolist(),normal_spread_degrees=normal_spread,
        fitted_plane_offset_m=float(fit.x[2]/np.linalg.norm(np.r_[-fit.x[:2],1.])),
        selected_plane_offset_m=float(-transform[2,3]),
        offset_selection='Normal from robust plane regression; offset is median inlier n dot point, not regression intercept',
        proposed_setup_from_source=transform.tolist(),aligned_initial_mesh_bottoms_m=aligned_bottoms,
        tilted_plane_quality_pass=bool(len(points)>1000 and inlier.mean()>.8 and np.median(residual)<.005 and normal_spread<2),
        quality_pass=bool(len(points)>1000 and inlier.mean()>.8 and tilt<5 and np.quantile(horizontal_error,.95)<.025),
        limitations=['Stereo and camera pose uncertainty remain; tabletop thickness4cm is an explicit simulation assumption.',
                    'This support-plane estimate does not alter source object trajectories or metric targets.'])
    (out/'table_support.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=int,nargs='+',required=True)
    for ep in p.parse_args().episodes:inspect(ep)
