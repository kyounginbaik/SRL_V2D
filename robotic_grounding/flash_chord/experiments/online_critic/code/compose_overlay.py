"""Actual-reference comparison: translucent original hands and synchronized crops."""
from pathlib import Path
import sys,json,subprocess,os
import numpy as np,pyarrow.parquet as pq,cv2,imageio_ffmpeg
B=Path(__file__).resolve().parents[1]
phase,ep,variant,seed,version=sys.argv[1:6];version=int(version)
root=B/'runs'/phase/ep/variant/f'seed_{seed}'
parent=root/'reference_overlays'/f'version_{version:04d}';out=parent/'transparent_closeups'
paths=[root/'training/references/reference_0000.npz',root/f'training/references/reference_{version:04d}.npz']
data=[np.load(p,allow_pickle=False) for p in paths]
ref=pq.read_table(B/'frozen'/ep/'reference.parquet').to_pylist()[0];frames=len(ref['object_body_position']);fps=float(ref['fps'])
meta=json.loads((out/'projection.json').read_text());matrix=np.array(meta['projection'])@np.array(meta['view']);W,H=meta['width'],meta['height']
def project(points):
 p=np.asarray(points);h=np.concatenate([p,np.ones(p.shape[:-1]+(1,))],-1)@matrix.T;n=h[...,:2]/h[...,3:4]
 return (n*np.array([1,-1])+1)*np.array([W,H])/2
wrists={s:project(np.stack([d[s+'/wrist_pos_w'] for d in data])) for s in ['left','right']}
objects=project(np.array(ref['object_body_position']))
cv2.setNumThreads(4)
caps=[cv2.VideoCapture(str(out/f'{x}.mp4')) for x in ['initial','current','initial_hands']]
assert all(c.isOpened() for c in caps)
# One fixed main crop for all frames and both reference versions.
bounds=[]
for cap in caps[:2]:
 while True:
  ok,f=cap.read()
  if not ok:break
  bg=np.median(f[:20,:20].reshape(-1,3),0);mask=(np.max(abs(f.astype(float)-bg),-1)>30).astype(np.uint8)
  n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
  for x,y,w,h,area in stats[1:]:
   if area>80:bounds.append([x,y,x+w,y+h])
 cap.set(cv2.CAP_PROP_POS_FRAMES,0)
a=np.array(bounds);lo=a[:,:2].min(0);hi=a[:,2:].max(0);center=(lo+hi)/2;main_height=max((hi-lo)[1],(hi-lo)[0]*.8)*1.14
# Fixed physical image scale for close-ups; center follows wrist/object midpoint.
# Estimate a fixed handle-region anchor in object-local coordinates from the
# reconstructed geometric contact labels. This sets the camera crop only.
from scipy.spatial.transform import Rotation
rot=Rotation.from_quat(np.array(ref['object_body_wxyz']).reshape(-1,4),scalar_first=True).as_matrix().reshape(frames,len(ref['object_body_names']),3,3)
pos=np.asarray(ref['object_body_position'])
centers={};close_heights={};object_indices={};closeup_anchors={}
for j,side in enumerate(['left','right']):
 ids=np.asarray(ref.get(f'mano_{side}_object_contact_part_ids',[]))
 counts=[np.count_nonzero(ids==o+1) for o in range(len(ref['object_body_names']))]
 if not counts or max(counts)==0:
  # Camera-only fallback: follow the actual refined wrist instead of
  # inventing a contact point or assuming this hand holds an object.
  centers[side]=wrists[side][1];close_heights[side]=180.;object_indices[side]=None
  closeup_anchors[side]={'mode':'refined_wrist','reason':'no_object_contact_labels'}
  print(f'[reference-overlay] {side} closeup follows refined wrist: no object contact labels',flush=True)
  continue
 cp=np.asarray(ref[f'mano_{side}_object_contact_positions'])
 j=int(np.argmax(counts));object_indices[side]=j
 local=np.einsum('tki,tij->tkj',cp-pos[:,j,None,:],rot[:,j])
 valid=ids==j+1
 anchor=np.median(local[valid],axis=0)
 world=np.einsum('tij,j->ti',rot[:,j],anchor)+pos[:,j]
 centers[side]=project(world);close_heights[side]=180.
 closeup_anchors[side]={'mode':'object_local_contact_median','object_name':ref['object_body_names'][j]}
def crop(image,center,height,size):
 ow,oh=size;hh=float(height);ww=hh*ow/oh
 transform=np.array([[ww/ow,0,center[0]-ww/2],[0,hh/oh,center[1]-hh/2]],np.float32)
 return cv2.warpAffine(image,transform,size,flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT,borderValue=(38,43,52))
def text(im,s,pos,scale=.65,color=(235,235,235)):
 cv2.putText(im,s,pos,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
temp=out/'overlay.partial.mp4'
proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1600x960','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','4','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],stdin=subprocess.PIPE)
for i in range(frames):
 images=[]
 for c in caps:
  ok,f=c.read();assert ok;images.append(f)
 old,cur,hands=images;bg=np.median(hands[:20,:20].reshape(-1,3),0)
 mask=np.clip((np.max(abs(hands.astype(float)-bg),-1)-15)/20,0,1)
 # Cyan shading preserves joint detail; opacity remains 45% even at overlap.
 gray=cv2.cvtColor(hands,cv2.COLOR_BGR2GRAY).astype(float)/255
 tint=np.array([255,218,55])[None,None,:]*(.35+.65*gray[...,None])
 alpha=.45*mask[...,None];blend=np.clip(cur*(1-alpha)+tint*alpha,0,255).astype(np.uint8)
 canvas=np.full((960,1600,3),(25,29,35),np.uint8)
 text(canvas,f'Original + refined reference v{version:04d}',(24,35),.85)
 text(canvas,'Cyan: original hands, 45% opacity  |  White: refined hands  |  Objects unchanged',(24,68),.6)
 canvas[100:900,:1000]=crop(blend,center,main_height,(1000,800))
 for j,side in enumerate(['left','right']):
  y=100+j*400;canvas[y+40:y+390,1010:1590]=crop(blend,centers[side][i],close_heights[side],(580,350))
  object_index=object_indices[side]
  label=(ref['object_body_names'][object_index].replace('_',' ')+' / '+side+' hand'
         if object_index is not None else side+' hand / wrist crop')
  text(canvas,label,(1025,y+27),.7)
  cv2.rectangle(canvas,(1010,y+40),(1590,y+390),(210,185,45),1)
 delta=[np.linalg.norm(data[1][s+'/wrist_pos_w'][i]-data[0][s+'/wrist_pos_w'][i])*100 for s in ['left','right']]
 text(canvas,f't={i/fps:.2f}s  |  Wrist change: left {delta[0]:.2f} cm, right {delta[1]:.2f} cm  |  Kinematic reference, not physics',(24,938),.62)
 proc.stdin.write(canvas.tobytes())
 if i in [0,120,200]:cv2.imwrite(str(out/f'preview_{i:03d}.jpg'),canvas)
proc.stdin.close();assert proc.wait()==0
for c in caps:c.release()
c=cv2.VideoCapture(str(temp));n=0
while c.read()[0]:n+=1
c.release();assert n==frames
temp.replace(out/'overlay.mp4')
manifest={'style':'transparent_closeups_v2','version':version,'frames':frames,'fps':fps,'opacity':.45,'original_snapshot':str(paths[0]),'refined_snapshot':str(paths[1]),'objects_tinted':False,'main_crop_fixed':True,'closeups':'same-camera crops tracking object-local median contact region or refined wrist when contact labels are absent, fixed 180-pixel height','closeup_anchors':closeup_anchors,'image_size':[1600,960]}
(out/'complete.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('OVERLAY_COMPLETE',out,flush=True)
