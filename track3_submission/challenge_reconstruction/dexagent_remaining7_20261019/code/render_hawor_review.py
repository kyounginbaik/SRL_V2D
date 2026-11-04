#!/usr/bin/env python3
"""CPU keypoint overlays of transferred HaWoR and exact Sharpa IK on the source RGB."""
from pathlib import Path
import argparse,json,hashlib,os
import numpy as np, cv2, pyarrow.parquet as pq
B=Path(__file__).resolve().parents[1]
CHAINS=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]
def project(points,C,K):
 cam=(points-C[:3,3])@C[:3,:3];p=cam@K.T
 return p[:,:2]/p[:,2:3],cam[:,2]>.001

def render(e):
 out=Path(os.environ.get('HAWOR_OUTPUT_ROOT',str(B/'hawor_recon')))/f'episode_{e:06d}';rgb=B/f'rgb/episode_{e:06d}/anycalib/undistorted.mp4'
 h=np.load(out/'hawor_decoded.npz');c=np.load(out/'camera_motion.npz');row=pq.ParquetFile(out/'reference.parquet').read().to_pylist()[0]
 n=len(h['frames']);times=sorted(set([0,n//4,n//2,3*n//4,n-1]));cap=cv2.VideoCapture(str(rgb));assert cap.isOpened(),rgb
 panels=[[],[],[]];report={'episode':e,'frames':times,'RGB':str(rgb),'video_sha256':hashlib.sha256(rgb.read_bytes()).hexdigest(),'camera_sha256':hashlib.sha256((out/'camera_motion.npz').read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256((out/'reference.parquet').read_bytes()).hexdigest(),'hand_source':'HaWoR; original WiLoR excluded','layout':['top: original RGB','middle: HaWoR MANO joints (orange left, blue right)','bottom: exact Sharpa IK wrist/MP/tips (same colors)'],'notes':['Held/interpolated hands marked FILLED; may be offscreen.','No object or source-image coordinate fitted from ground truth.']}
 colors={'left':(30,150,255),'right':(255,170,30)}
 for frame in times:
  cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();assert ok
  images=[im.copy(),im.copy(),im.copy()]
  C=c['world_from_rgb'][frame];K=c['K']
  for side in ['left','right']:
   color=colors[side];uv,valid=project(h[side+'_joints_world'][frame],C,K)
   for chain in CHAINS:
    for a,b in zip(chain,chain[1:]):
     if valid[a] and valid[b]:cv2.line(images[1],tuple(uv[a].astype(int)),tuple(uv[b].astype(int)),color,6,cv2.LINE_AA)
   for point,v in zip(uv,valid):
    if v:cv2.circle(images[1],tuple(point.astype(int)),7,color,-1,cv2.LINE_AA)
   names=row[side+'_robot_frame_names'];fk=np.array(row['robot_'+side+'_frames'][frame]);inds=[names.index(side+'_hand_C_MC')]
   for finger,mid in [('thumb','MCP_VL'),('index','MP'),('middle','MP'),('ring','MP'),('pinky','MP')]:inds += [names.index(f'{side}_{finger}_{mid}_site'),names.index(f'{side}_{finger}_tip_site')]
   uv,v=project(fk[inds,:3],C,K)
   for i in range(1,11,2):
    for a,b in [(0,i),(i,i+1)]:
     if v[a] and v[b]:cv2.line(images[2],tuple(uv[a].astype(int)),tuple(uv[b].astype(int)),color,6,cv2.LINE_AA)
   for point,visible in zip(uv,v):
    if visible:cv2.circle(images[2],tuple(point.astype(int)),9,color,-1,cv2.LINE_AA)
  for r,img in enumerate(images):
   panel=cv2.resize(img,(384,288));header=np.zeros((35,384,3),np.uint8)
   title=['ORIGINAL RGB','HaWoR MANO','Sharpa IK'][r]+f' | {frame/20:.2f}s'
   cv2.putText(header,title,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
   if r and any(not h[s+'_valid'][frame] for s in ['left','right']):cv2.putText(panel,'FILLED / UNOBSERVED HAND',(8,278),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1,cv2.LINE_AA)
   panels[r].append(np.vstack([header,panel]))
 cap.release();sheet=np.vstack([np.hstack(row) for row in panels]);target=out/'hawor_keypoint_review.jpg';assert cv2.imwrite(str(target),sheet)
 (out/'hawor_keypoint_review.json').write_text(json.dumps(report,indent=2)+'\n');print(target,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('episodes',nargs='+',type=int);a=p.parse_args()
 for e in a.episodes:render(e)
