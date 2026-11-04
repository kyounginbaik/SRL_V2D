import sys,json,glob,os
sys.path.insert(0,'.')
import numpy as np,torch,cv2
from hawor.utils.process import run_mano,run_mano_left
Z='/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103/';V='/scratch2/exaFLOPs/srl_yam_run/source_videos/ego_cam_a/'
tt=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32).cuda()[None]
rows=[]
from scipy.spatial.transform import Rotation as R
for n in map(int,sys.argv[1:]):
    e=f'episode_{n:06d}';f=sorted(glob.glob(f'../root/hands/{e}/*.npz'))[0];d=np.load(f);W=d['camera_to_world_transform'];I=json.load(open(Z+f'bundles/{e}/intrinsics.json'));fx,cx,cy=2*I['fx'],2*I['cx'],2*I['cy'];T=len(W);J={}
    o=np.load(Z+f'hands/{e}/'+os.path.basename(f));sc=2/(float(o['hand_left_scale'].mean())+float(o['hand_right_scale'].mean()));C=o['camera_to_world_transform'].astype(float);Rm=R.from_matrix(np.array([W[i,:3,:3]@C[i,:3,:3].T for i in range(T)])).mean();tr=np.median(W[:,:3,3]-Rm.apply(C[:,:3,3]*sc),axis=0)
    for s in('left','right'):
        X=d[f'hand_{s}_wrist_to_world_transform'].astype(float);rot=R.from_matrix(X[:,:3,:3]).as_rotvec();fn=run_mano if s=='right' else run_mano_left
        b=np.tile(d[f'hand_{s}_betas'],(T,1));J[s]=fn(tt(X[:,:3,3]),tt(rot),tt(d[f'hand_{s}_finger_pose'].reshape(T,45)),betas=tt(b))['joints'][0].cpu().numpy()
        ow=Rm.apply(o[f'hand_{s}_wrist_to_world_transform'][:,:3,3].astype(float)*sc)+tr;dd=np.linalg.norm(ow-X[:,:3,3],axis=1)*100;ob=d[f'hand_{s}_is_valid'].astype(bool);print(n,s,'old-vs-new anchor cm median',round(float(np.nanmedian(dd[ob])),1) if ob.any() else None,'p90',round(float(np.nanpercentile(dd[ob],90)),1) if ob.any() else None,'wrist joint vs anchor cm',round(float(np.median(np.linalg.norm(J[s][:,0]-X[:,:3,3],axis=1))*100),1),flush=True)
    c=cv2.VideoCapture(V+e+'.mp4');tl=[]
    for fr in np.linspace(0,T-1,6).astype(int):
        c.set(cv2.CAP_PROP_POS_FRAMES,int(fr));ok,im=c.read()
        for s,col in(('left',(255,120,0)),('right',(0,0,255))):
            ob=bool(d[f'hand_{s}_is_valid'][fr]);q=(J[s][fr]-W[fr,:3,3])@W[fr,:3,:3];
            for x,y,z in q:
                if z>0:
                    u,v=fx*x/z+cx,fx*y/z+cy
                    if 0<=u<2028 and 0<=v<1520:cv2.circle(im,(int(u),int(v)),10,col,-1 if ob else 3)
        cv2.putText(im,f'ep{n} f{fr}',(30,110),0,3,(255,255,255),8);tl.append(cv2.resize(im,(338,253)))
    rows.append(np.hstack(tl))
    if len(rows)==4 or n==int(sys.argv[-1]):cv2.imwrite(f'../out/sheet_{n}.png',np.vstack(rows));rows=[]
print('SHEETS_DONE')
