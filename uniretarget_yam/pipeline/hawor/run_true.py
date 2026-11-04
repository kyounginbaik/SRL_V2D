"""HaWoR with the recording's true focal length and true camera path instead of an estimated focal and SLAM.
Frames are kept 1:1 with the source video (the stock extractor resamples to 30 fps)."""
import argparse,json,os,shutil,subprocess,sys
sys.path.insert(0,'.')
import numpy as np,torch
from scipy.spatial.transform import Rotation as R
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_video import hawor_motion_estimation,hawor_infiller
p=argparse.ArgumentParser();p.add_argument('--episode',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
Z='/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103/bundles/';e=f'episode_{a.episode:06d}';src=f'/scratch2/exaFLOPs/srl_yam_run/source_videos/ego_cam_a/{e}.mp4'
I=json.load(open(Z+e+'/intrinsics.json'));b=np.load(Z+e+'/object_poses.npz');W=b['world_from_rgb'];focal=2*I['fx']
work=os.path.join(a.out,e);os.makedirs(work,exist_ok=True);vid=os.path.join(work,e+'.mp4');shutil.copy(src,vid) if not os.path.exists(vid) else None
seq=os.path.join(work,e);imgs=os.path.join(seq,'extracted_images');os.makedirs(imgs,exist_ok=True)
if not os.listdir(imgs):subprocess.run(['ffmpeg','-y','-loglevel','error','-i',vid,'-vsync','0','-start_number','0',os.path.join(imgs,'%04d.jpg')],check=True)
args=argparse.Namespace(video_path=vid,input_type='file',img_focal=focal,checkpoint='./weights/hawor/checkpoints/hawor.ckpt',infiller_weight='./weights/hawor/checkpoints/infiller.pt',vis_mode='world')
n=len(os.listdir(imgs));assert n==len(W),(n,len(W))
start,end,seq2,files=detect_track_video(args);assert os.path.abspath(seq2)==os.path.abspath(seq),(seq2,seq);print('frames',n,'range',start,end,'focal',focal,flush=True)
open(os.path.join(seq,'est_focal.txt'),'w').write(str(focal))
# The wearer's hands fill far more of this camera than anyone else's: drop tracks much smaller than the hand tracks of the same episode.
tp=os.path.join(seq,f'tracks_{start}_{end}/model_tracks.npy');tracks=np.load(tp,allow_pickle=True).item();area=lambda v:float(np.median([(x['det_box'][0][2]-x['det_box'][0][0])*(x['det_box'][0][3]-x['det_box'][0][1]) if np.ndim(x['det_box'])==2 else (x['det_box'][2]-x['det_box'][0])*(x['det_box'][3]-x['det_box'][1]) for x in v if x['det']]))
top=max(area(v) for v in tracks.values());dropped=[(float(k),len(v),round(area(v))) for k,v in tracks.items() if area(v)<.3*top];kept={k:v for k,v in tracks.items() if area(v)>=.3*top};np.save(tp,kept);print('tracks kept',[(float(k),v[0]['frame'],v[-1]['frame'],round(area(v))) for k,v in kept.items()],'dropped',dropped,flush=True)
observed=np.zeros((2,n),bool)
for v in kept.values():
    v_=[x for x in v if x['det']];right=np.concatenate([x['det_handedness'] for x in v_]).mean()>=.5
    for x in v_:observed[int(right),int(x['frame'])]=True
cf=os.path.join(seq,f'tracks_{start}_{end}/frame_chunks_all.npy');os.path.exists(cf) and os.remove(cf)
chunks,f=hawor_motion_estimation(args,start,end,seq);assert abs(f-focal)<1e-3,f
os.makedirs(os.path.join(seq,'SLAM'),exist_ok=True);traj=np.concatenate([W[:,:3,3],R.from_matrix(W[:,:3,:3]).as_quat()],1).astype(np.float32)
np.savez(os.path.join(seq,f'SLAM/hawor_slam_w_scale_{start}_{end}.npz'),traj=traj,scale=np.float32(1.0),tstamp=np.arange(n),img_focal=focal,img_center=np.array([I['cx'],I['cy']])*2)
nn=lambda x:x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
tr,rot,pose,betas,valid=hawor_infiller(args,start,end,chunks)
np.savez(os.path.join(a.out,f'{e}_hawor_true.npz'),trans=nn(tr),rot=nn(rot),hand_pose=nn(pose),betas=nn(betas),valid=nn(valid),world_from_rgb=W,focal=focal,observed=observed,hand_order=np.array(['left','right']))
print('HAWOR_TRUE_DONE',e,tr.shape,flush=True)
