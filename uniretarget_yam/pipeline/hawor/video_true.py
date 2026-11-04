"""HaWoR hand mesh over the source video. Blue left, red right; grey outline and a label when the frame was filled in, not detected."""
import sys,json,glob,os,subprocess
sys.path.insert(0,'.')
import numpy as np,torch,cv2
from scipy.spatial.transform import Rotation as R
from hawor.utils.process import run_mano,run_mano_left,get_mano_faces
Z='/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103/';V='/scratch2/exaFLOPs/srl_yam_run/source_videos/ego_cam_a/';O='/scratch2/exaFLOPs/hawor_rerun/videos/';os.makedirs(O,exist_ok=True)
tt=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32).cuda()[None]
F=np.asarray(get_mano_faces()).astype(int)
for n in map(int,sys.argv[1:]):
    e=f'episode_{n:06d}';d=np.load(sorted(glob.glob(f'../root/hands/{e}/*.npz'))[0]);W=d['camera_to_world_transform'].astype(float);I=json.load(open(Z+f'bundles/{e}/intrinsics.json'));fx,fy,cx,cy=I['fx'],I['fy'],I['cx'],I['cy'];T=len(W);G={};ob={}
    for s in('left','right'):
        X=d[f'hand_{s}_wrist_to_world_transform'].astype(float);fn=run_mano if s=='right' else run_mano_left;G[s]=fn(tt(X[:,:3,3]),tt(R.from_matrix(X[:,:3,:3]).as_rotvec()),tt(d[f'hand_{s}_finger_pose'].reshape(T,45)),betas=tt(np.tile(d[f'hand_{s}_betas'],(T,1))))['vertices'][0].cpu().numpy().astype(float);ob[s]=d[f'hand_{s}_is_valid'].astype(bool)
    cap=cv2.VideoCapture(V+e+'.mp4');tmp=O+f'.{e}.mp4';wr=cv2.VideoWriter(tmp,cv2.VideoWriter_fourcc(*'mp4v'),20,(1014,760))
    for f in range(T):
        ok,im=cap.read()
        if not ok:break
        im=cv2.resize(im,(1014,760));ov=im.copy();y=70
        for s,col in(('left',(255,130,30)),('right',(40,40,235))):
            q=(G[s][f]-W[f,:3,3])@W[f,:3,:3]
            if (q[:,2]>.05).all():
                uv=np.stack([fx*q[:,0]/q[:,2]+cx,fy*q[:,1]/q[:,2]+cy],1);z=q[F][:,:,2].mean(1);c=(col if ob[s][f] else (170,170,170))
                for t in F[np.argsort(-z)]:cv2.fillConvexPoly(ov,np.round(uv[t]).astype(np.int32),c,lineType=cv2.LINE_AA)
            cv2.putText(im,f"{s}: {'detected' if ob[s][f] else 'FILLED IN'}",(16,y),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255) if ob[s][f] else (0,220,255),2,cv2.LINE_AA);y+=30
        im=cv2.addWeighted(ov,.55,im,.45,0);cv2.putText(im,f"ep{n} frame {f}/{T-1}  HaWoR, true camera",(16,36),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2,cv2.LINE_AA);wr.write(im)
    wr.release();out=O+f'ep{n:02d}_HaWoR_True_Camera.mp4';subprocess.run(['ffmpeg','-y','-loglevel','error','-i',tmp,'-c:v','libx264','-crf','20','-pix_fmt','yuv420p',out],check=True);os.remove(tmp);print('VIDEO',out,flush=True)
print('VIDEOS_DONE')
