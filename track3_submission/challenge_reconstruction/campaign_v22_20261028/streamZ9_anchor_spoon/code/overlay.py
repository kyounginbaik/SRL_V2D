"""Overlay: projected spoon mesh points (green), SAM2 mask contour (red), ball detections (blue) and model balls (yellow)."""
import sys, json, cv2, numpy as np, trimesh
from pathlib import Path
R=Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
ep,t=int(sys.argv[1]),int(sys.argv[2]); out=sys.argv[3]
root=R/f'canonical/evaluation/episode_{ep:06d}'; fol=Path(sys.argv[4]) if len(sys.argv)>4 else R/f'selected20_final_v5/episode_{ep:06d}'
cap=cv2.VideoCapture(str(root/'rgb.mp4')); cap.set(cv2.CAP_PROP_POS_FRAMES,t); ok,img=cap.read()
intr=json.loads((root/'intrinsics.json').read_text()); K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
if img.shape[1]!=intr['width']: img=cv2.resize(img,(intr['width'],intr['height']))
d=np.load(fol/'object_poses.npz'); s=[str(n) for n in d['names']].index('wooden_spoon'); P=d['camera_from_object'][t,s]
m=trimesh.load(fol/'meshes/wooden_spoon.obj',force='mesh',process=False); pts=np.asarray(trimesh.sample.sample_surface(m,20000,seed=1)[0])
c=pts@P[:3,:3].T+P[:3,3]; uv=(c[:,:2]/c[:,2:])@K[:2,:2].T+K[:2,2]
for u,v in uv.astype(int): 
    if 0<=u<img.shape[1] and 0<=v<img.shape[0]: img[v,u]=(0,255,0)
mask=cv2.imread(str(root/'objects/wooden_spoon/masks/1'/f'{t:06d}.png'),0)
cs,_=cv2.findContours((mask>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(img,cs,-1,(0,0,255),1)
model=np.array(json.loads((root/'exports/spoon_roll_v8/selection.json').read_text())['objects']['wooden_spoon']['ball_roll']['model_body_m'])
mc=model@P[:3,:3].T+P[:3,3]; muv=(mc[:,:2]/mc[:,2:])@K[:2,:2].T+K[:2,2]
for u,v in muv: cv2.circle(img,(int(u),int(v)),4,(0,255,255),1)
for b in json.loads((root/'ball_detections_wooden_spoon.json').read_text())['frames'][t]: cv2.circle(img,tuple(int(x) for x in b['uv']),7,(255,0,0),1)
x0,y0=np.maximum(uv.min(0).astype(int)-40,0); x1,y1=uv.max(0).astype(int)+40
crop=img[y0:y1,x0:x1]; crop=cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST); cv2.imwrite(out,crop)
