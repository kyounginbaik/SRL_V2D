"""Draw new (true camera) and old HaWoR hand joints on the source video and compare."""
import sys,json,os,glob
sys.path.insert(0,'.')
import numpy as np,torch,cv2
from hawor.utils.process import run_mano,run_mano_left
Z='/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103/';V='/scratch2/exaFLOPs/srl_yam_run/source_videos/ego_cam_a/'
def joints(tr,rot,pose,bet,side):
    f=run_mano if side=='right' else run_mano_left
    t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32).cuda()[None]
    return f(t(tr),t(rot),t(pose),betas=t(bet))['joints'][0].cpu().numpy()
def proj(J,W,fx,cx,cy):
    # J world joints, W world_from_cam
    p=(J-W[:,None,:3,3]);c=np.einsum('tij,tkj->tki',W[:,:3,:3].transpose(0,2,1),p);return np.stack([fx*c[...,0]/c[...,2]+cx,fx*c[...,1]/c[...,2]+cy],-1),c
for n in map(int,sys.argv[1:]):
    e=f'episode_{n:06d}';d=np.load(f'../out/{e}_hawor_true.npz');I=json.load(open(Z+f'bundles/{e}/intrinsics.json'));W=d['world_from_rgb'];fx,cx,cy=2*I['fx'],2*I['cx'],2*I['cy'];T=len(W)
    b=np.load(Z+f'bundles/{e}/object_poses.npz');names=[str(x) for x in b['names']]
    new={};
    for h,s in enumerate(('left','right')):
        new[s]=joints(d['trans'][h],d['rot'][h],d['hand_pose'][h],d['betas'][h],s)
        w=new[s][:,0];st=np.linalg.norm(np.diff(w,axis=0),axis=1)*100;c=np.einsum('tij,tj->ti',W[:,:3,:3].transpose(0,2,1),w-W[:,:3,3])
        import trimesh
        D=[]
        for k,nm in enumerate(names):
            v=np.asarray(trimesh.load(Z+f'targets/{e}/{nm}_mesh_metric.obj',force='mesh').vertices)[::20];TT=np.load(Z+f'targets/{e}/{nm}_trajectory.npz')['T_world_obj'];D.append([np.linalg.norm((TT[f,:3,:3]@v.T).T+TT[f,:3,3]-new[s][f,9],axis=1).min()*100 for f in range(T)])
        print(n,s,'valid',round(float(d['valid'][h].mean()),2),'step cm p50/p99/max',np.percentile(st,[50,99]).round(1),round(st.max(),1),'jumps>25',int((st>25).sum()),'depth m p5/50/95',np.percentile(c[:,2],[5,50,95]).round(2),'palm-obj cm min',[round(min(x),1) for x in D],flush=True)
    o=np.load(sorted(glob.glob(Z+f'hands/{e}/hawor_*_result.npz'))[0]);ofx,ofy,ocx,ocy=np.array(o['camera_intrinsics']).reshape(-1)[:4]
    cap=cv2.VideoCapture(V+e+'.mp4');tiles=[]
    for f in np.linspace(0,T-1,8).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(f));ok,im=cap.read()
        for s,col in (('left',(255,120,0)),('right',(0,0,255))):
            uv,_=proj(new[s][f:f+1],W[f:f+1],fx,cx,cy)
            for u,v in uv[0]:
                if 0<=u<2028 and 0<=v<1520:cv2.circle(im,(int(u),int(v)),9,col,-1)
            t=o[f'hand_{s}_wrist_trans_in_camera'][f].astype(float);cv2.drawMarker(im,(int(ofx*t[0]/t[2]+ocx),int(ofy*t[1]/t[2]+ocy)),(255,255,255),cv2.MARKER_TILTED_CROSS,50,6)
        cv2.putText(im,f'f{f}',(40,110),0,3,(255,255,255),8);tiles.append(cv2.resize(im,(507,380)))
    cv2.imwrite(f'../out/{e}_compare.png',np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])]))
print('EVAL_DONE')
