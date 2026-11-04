"""The zip stores left-hand MANO parameters in right-hand space (negated y,z axis-angle), which its loader undoes; HaWoR gives native left, so they are flipped here.
Write the rerun as hawor_<object>_result.npz in the schema of the reconstruction's hand files, in the bundle world at metric scale."""
import glob,json,os,sys
import numpy as np
from scipy.spatial.transform import Rotation as R,Slerp
Z='/scratch2/exaFLOPs/latest_recon/v2/best_reconstruction_20261103/';O='/scratch2/exaFLOPs/hawor_rerun/root/';rep=[]
for f in sorted(glob.glob('/scratch2/exaFLOPs/hawor_rerun/out/episode_*_hawor_true.npz')):
    e=os.path.basename(f)[:14];d=dict(np.load(f));W=d['world_from_rgb'];n=len(W);I=json.load(open(Z+f'bundles/{e}/intrinsics.json'));os.makedirs(O+f'hands/{e}',exist_ok=True);out0={};row=dict(episode=e)
    for h,s in enumerate(('left','right')):
        t=d['trans'][h].astype(float);r=d['rot'][h].astype(float);p=d['hand_pose'][h].astype(float);b=d['betas'][h].astype(float);ob=d['observed'][h].copy();idx=np.flatnonzero(ob)
        if len(idx)==0:row[s]='never detected';idx=np.array([0])
        # The infiller leaves nothing before the first and after the last detection: hold the nearest detected pose there.
        for fr in list(range(0,idx[0]))+list(range(idx[-1]+1,n)):
            k=idx[0] if fr<idx[0] else idx[-1];t[fr]=t[k];r[fr]=r[k];p[fr]=p[k];b[fr]=b[k]
        Rw=R.from_rotvec(r);T=np.tile(np.eye(4),(n,1,1));T[:,:3,:3]=Rw.as_matrix();T[:,:3,3]=t;Wi=np.linalg.inv(W);Tc=Wi@T
        out0.update({f'hand_{s}_betas':b[idx].mean(0).astype(np.float32),f'hand_{s}_finger_pose':(p.reshape(n,15,3)*(np.array([1,-1,-1.]) if s=='left' else 1)).astype(np.float32),f'hand_{s}_scale':np.ones(n,np.float32),f'hand_{s}_is_valid':ob,f'hand_{s}_wrist_to_world_transform':T.astype(np.float32),f'hand_{s}_wrist_to_camera_transform':Tc.astype(np.float32),f'hand_{s}_wrist_trans_in_camera':Tc[:,:3,3].astype(np.float32),f'hand_{s}_wrist_orient_in_camera':(R.from_matrix(Tc[:,:3,:3]).as_rotvec()*(np.array([1,-1,-1.]) if s=='left' else 1)).astype(np.float32)})
        st=np.linalg.norm(np.diff(t,axis=0),axis=1)*100;row[s+'_observed']=round(float(ob.mean()),2);row[s+'_jumps']=int((st>25).sum());row[s+'_maxstep']=round(float(st.max()),1);row[s+'_depth_min']=round(float(Tc[:,2,3].min()),2)
    for g in sorted(glob.glob(Z+f'hands/{e}/hawor_*_result.npz')):
        o=np.load(g);name=os.path.basename(g)[6:-11];TT=np.load(Z+f'targets/{e}/{name}_trajectory.npz')['T_world_obj'];res=dict(o);res.update(out0)
        # Same world as the original hand files, gravity-aligned, only metric: every consumer takes up and heading from it.
        sc=2/(float(o['hand_left_scale'].mean())+float(o['hand_right_scale'].mean()));C=o['camera_to_world_transform'].astype(float);Rm=R.from_matrix(np.array([W[i,:3,:3]@C[i,:3,:3].T for i in range(n)])).mean();tr=np.median(W[:,:3,3]-Rm.apply(C[:,:3,3]*sc),axis=0);S=np.eye(4);S[:3,:3]=Rm.as_matrix();S[:3,3]=tr;Si=np.linalg.inv(S)
        res['camera_to_world_transform']=(Si@W).astype(np.float32);res['object_to_world_transform']=(Si@TT).astype(np.float32);res['object_to_camera_transform']=(np.linalg.inv(W)@TT).astype(np.float32)
        for s_ in('left','right'):res[f'hand_{s_}_wrist_to_world_transform']=(Si@out0[f'hand_{s_}_wrist_to_world_transform'].astype(float)).astype(np.float32)
        res['camera_intrinsics']=np.array([2*I['fx'],2*I['fy'],2*I['cx'],2*I['cy']],np.float32);np.savez(O+f'hands/{e}/{os.path.basename(g)}',**res)
    rep.append(row);print(row)
for sub in ('bundles','targets','submitted_meshes'):
    if not os.path.exists(O+sub):os.symlink(Z+sub,O+sub)
print('CONVERT_DONE',len(rep))
