from pathlib import Path
import sys,json,csv,subprocess,os,shutil
import numpy as np,pyarrow.parquet as pq
B=Path(__file__).resolve().parent.parent
phase,ep,variant,seed,version=sys.argv[1:6];version=int(version)
root=B/'runs'/phase/ep/variant/f'seed_{seed}'
out=root/'reference_overlays'/f'version_{version:04d}'
paths=[root/'training/references/reference_0000.npz',root/f'training/references/reference_{version:04d}.npz'];data=[np.load(p,allow_pickle=False) for p in paths]
reference=pq.read_table(B/'frozen'/ep/'reference.parquet').to_pylist()[0]
frames=len(reference['object_body_position']);fps=float(reference['fps']);chunk_frames=round(fps)
pos=np.array(reference['object_body_position']).reshape(-1,3);center=np.median(pos,axis=0);radius=max(.5,float(np.percentile(np.linalg.norm(pos-center,axis=1),95))+.35);cam=center+radius*np.array([1.8,-2.3,1.7])
import cv2,imageio_ffmpeg
cv2.setNumThreads(4)
caps=[cv2.VideoCapture(str(out/f'{x}.mp4')) for x in ['initial','current']]
# Fit a single crop to the complete initial/current sequences, keeping motion visible.
# Use both views so edits cannot be hidden by a crop chosen from just one reference.
x0,y0,x1,y1=960,540,0,0
for cap in caps:
 while True:
  ok,frame=cap.read()
  if not ok:break
  bg=np.median(frame[:20,:20].reshape(-1,3),axis=0)
  mask=(np.max(np.abs(frame.astype(float)-bg),axis=-1)>35).astype(np.uint8)
  n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
  for x,y,w,h,area in stats[1:]:
   if area<30:continue
   x0=min(x0,x);y0=min(y0,y);x1=max(x1,x+w);y1=max(y1,y+h)
 cap.set(cv2.CAP_PROP_POS_FRAMES,0)
cx,cy=(x0+x1)/2,(y0+y1)/2
cw,ch=(x1-x0)*1.12,(y1-y0)*1.12
cw=max(cw,ch*960/540);ch=cw*540/960
scale=min(1.,960/cw,540/ch);cw=int(cw*scale);ch=int(ch*scale)
x0=int(np.clip(cx-cw/2,0,960-cw));y0=int(np.clip(cy-ch/2,0,540-ch))
def close_view(frame):return cv2.resize(frame[y0:y0+ch,x0:x0+cw],(960,540),interpolation=cv2.INTER_CUBIC)
(out/'framing.json').write_text(json.dumps({'crop_xywh':[x0,y0,cw,ch],'source_wh':[960,540],'method':'fixed full-sequence foreground union, initial and current, 12 percent padding'}))
p=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','bgr24','-s','960x600','-r','20','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'overlay.mp4')],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
for i in range(frames):
 pair=[]
 for c in caps:
  ok,f=c.read();assert ok;pair.append(close_view(f))
 old,cur=pair;bg=np.median(cur[:30,:30].reshape(-1,3),axis=0)
 oldmask=np.max(np.abs(old.astype(float)-bg),axis=-1)>28
 curmask=np.max(np.abs(cur.astype(float)-bg),axis=-1)>28
 alpha=oldmask[...,None]*.65
 tint=old*.25+np.array([255,220,60])*.75
 f=np.clip(cur*(1-alpha)+tint*alpha,0,255).astype(np.uint8)
 # A white current silhouette remains legible under the overlapping cyan ghost.
 current_edge=cv2.morphologyEx(curmask.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
 f[current_edge]=(245,245,245)
 canvas=np.zeros((600,960,3),np.uint8);canvas[60:]=f
 cv2.putText(canvas,f'Reference v{version:04d}{" (unchanged)" if version == 0 else ""} | White outline: current | Cyan 65%: original',(15,24),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
 deltas=[float(np.linalg.norm(data[1][side+'/wrist_pos_w'][i]-data[0][side+'/wrist_pos_w'][i])*100) for side in ['left','right']]
 cv2.putText(canvas,f't={i/fps:.2f}s | wrist change L={deltas[0]:.2f} cm R={deltas[1]:.2f} cm | true-scale kinematic reference',(15,48),cv2.FONT_HERSHEY_SIMPLEX,.48,(220,220,220),1,cv2.LINE_AA)
 p.stdin.write(canvas.tobytes())
p.stdin.close();assert p.wait()==0
for c in caps:c.release()
changes={'version':version,'frames':frames,'initial_snapshot':str(paths[0]),'current_snapshot':str(paths[1]),'provenance':json.loads(str(data[1]['header']))['provenance']}
rows=[]
for i in range(frames):
 row={'frame':i,'time_s':i/fps,'chunk':i//chunk_frames}
 for side in ['left','right']:
  row[side+'_wrist_delta_cm']=float(np.linalg.norm(data[1][side+'/wrist_pos_w'][i]-data[0][side+'/wrist_pos_w'][i])*100)
 rows.append(row)
with (out/'changes_over_time.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for side in ['left','right']:
 a=[r[side+'_wrist_delta_cm'] for r in rows];changes[side]={'mean_wrist_delta_cm':float(np.mean(a)),'max_wrist_delta_cm':float(np.max(a))}
(out/'changes.json').write_text(json.dumps(changes,indent=2))
url=os.path.relpath(B/'player',out)+'/?playbackPath='+os.path.relpath(out/'overlay.viser',B/'player')+'&initialCameraPosition='+','.join(map(str,cam))+'&initialCameraLookAt='+','.join(map(str,center))+'&initialCameraUp=0,0,1'
(out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h2>Initial ghost / current reference</h2><p>White: current. Transparent blue: fixed initialization. Kinematic playback.</p><a href="'+url+'">Interactive 3D</a><video controls width="960" src="overlay.mp4"></video><a href="changes_over_time.csv">Per-frame wrist changes</a>')
c=cv2.VideoCapture(str(out/'overlay.mp4'));count=0
while c.read()[0]:count+=1
assert count==frames
(out/'complete.json').write_text(json.dumps(changes,indent=2))
print('OVERLAY_COMPLETE',out,flush=True)
