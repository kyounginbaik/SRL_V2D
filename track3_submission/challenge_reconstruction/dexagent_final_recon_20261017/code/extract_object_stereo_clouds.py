"""Export observed masked stereo surfaces in the unchanged source world frame."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
B=Path(__file__).resolve().parents[1]

def extract(episode,frames,out):
    source=B.parent/f'final_recon/episode_{episode:06d}';out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((source/'config.json').read_text());cal=json.loads((source/'stereo_calibration/rgb_intrinsics.json').read_text())
    camera=np.load(source/'camera_motion.npz')['world_from_rgb'];poses=np.load(source/'object_poses.npz')
    capture=cv2.VideoCapture(str(Path(cfg['source'])/'anycalib/undistorted.mp4'));reports=[]
    for frame in frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,rgb=capture.read()
        if not ok:raise ValueError(f'missing RGB frame{frame}')
        h,w=rgb.shape[:2];K=np.array([[cal['fx'],0,cal['cx']],[0,cal['fy'],cal['cy']],[0,0,1.]])
        K[0]*=w/cal['width'];K[1]*=h/cal['height']
        raw=cv2.imread(str(Path(cfg['depth'])/f'{frame:06d}.png'),-1)
        with np.errstate(divide='ignore',invalid='ignore'):depth=65535/raw.astype(float)-1
        depth=cv2.resize(depth,(w,h),interpolation=cv2.INTER_NEAREST)
        hand=np.zeros((h,w),np.uint8)
        for track in cfg['hand_tracks']:
            hm=cv2.imread(str(Path(cfg['source'])/'masks'/track/f'{frame:06d}.png'),0)
            if hm is not None:hand|=cv2.resize(hm,(w,h),interpolation=cv2.INTER_NEAREST)
        hand=cv2.dilate(hand,np.ones((7,7),np.uint8))>0
        for name,folder in cfg['object_masks'].items():
            mask=cv2.imread(str(Path(folder)/f'{frame:06d}.png'),0)
            mask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
            mask=(cv2.erode(mask,np.ones((7,7),np.uint8))>0)&~hand
            y,x=np.where(mask&np.isfinite(depth)&(depth>.1)&(depth<2));z=depth[y,x]
            pc=np.c_[(x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z]
            world=pc@camera[frame,:3,:3].T+camera[frame,:3,3]
            target=out/f'{name}_{frame:06d}.npz'
            oi=list(poses['names'].astype(str)).index(name)
            np.savez_compressed(target,points_source=world,pixels=np.c_[x,y],rgb=rgb[y,x],K=K,world_from_rgb=camera[frame],world_from_object=poses['world_from_object'][frame,oi])
            reports.append(dict(object=name,frame=frame,points=len(world),path=str(target)))
        cv2.imwrite(str(out/f'rgb_{frame:06d}.jpg'),rgb)
        print('extracted',frame,flush=True)
    capture.release();(out/'manifest.json').write_text(json.dumps(dict(source=str(source),scope='observations only; hand pixels excluded; no pose/asset edits',records=reports),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--frames',type=int,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    extract(**vars(p.parse_args()))
