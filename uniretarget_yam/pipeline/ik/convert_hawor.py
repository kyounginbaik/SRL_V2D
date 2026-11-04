"""Write an ego_recon result.npz from the zip's HaWoR file and metric spoon target.
The HaWoR world is scaled; the driver multiplies world anchors by scene_scale, so the spoon is written divided by it."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
p=argparse.ArgumentParser();p.add_argument('--frames',type=int,default=0);p.add_argument('--pad',type=int,default=0);p.add_argument('--hawor',required=True);p.add_argument('--target',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reanchor',action='store_true');p.add_argument('--hold',type=int,nargs=2,default=None);p.add_argument('--side',default='right');p.add_argument('--up',required=True);p.add_argument('--max_speed',type=float,default=4.0);a=p.parse_args()
h=dict(np.load(a.hawor));T=np.load(a.target)['T_world_obj'];n=len(T)
scale=2.0/(float(h['hand_left_scale'].mean())+float(h['hand_right_scale'].mean()))
spikes={}
for sd in ('left','right'):
    W0=h[f'hand_{sd}_wrist_to_world_transform'].astype(float);sp=np.linalg.norm(np.diff(W0[:,:3,3]*scale,axis=0),axis=1)*20
    bad=np.zeros(len(W0),bool)
    for i in np.flatnonzero(sp>a.max_speed):
        bad[i]=bad[i] or (i>0 and sp[i-1]>a.max_speed);bad[i+1]=True
    bad[0]=False;idx=np.flatnonzero(~bad);spikes[sd]=np.flatnonzero(bad).tolist()
    if bad.any():
        f=np.arange(len(W0));W1=W0.copy()
        for c in range(3):W1[:,c,3]=np.interp(f,idx,W0[idx,c,3])
        from scipy.spatial.transform import Slerp
        W1[:,:3,:3]=Slerp(idx,R.from_matrix(W0[idx,:3,:3]))(f).as_matrix();h[f'hand_{sd}_wrist_to_world_transform']=W1.astype(np.float32)
        Oc=h[f'hand_{sd}_wrist_orient_in_camera'].astype(float);h[f'hand_{sd}_wrist_orient_in_camera']=Slerp(idx,R.from_rotvec(Oc[idx]))(f).as_rotvec().astype(np.float32)
        Fp=h[f'hand_{sd}_finger_pose'].astype(float).copy()
        for j in range(Fp.shape[1]):
            for c in range(3):Fp[:,j,c]=np.interp(f,idx,Fp[idx,j,c])
        h[f'hand_{sd}_finger_pose']=Fp.astype(np.float32)
O=h['object_to_world_transform'].astype(float);om=O[:,:3,3]*scale
# Rigid map from target world to metric HaWoR world, fitted on spoon positions.
tp=T[:,:3,3]
up=np.array(json.load(open(a.up))) if a.up else None
R1=R.align_vectors([[0,0,1.]],[up])[0]
d1=R1.apply(tp-tp[0]);d2=om-om[0]
num=float(np.sum(d2[:,1]*d1[:,0]-d2[:,0]*d1[:,1]));den=float(np.sum(d2[:,0]*d1[:,0]+d2[:,1]*d1[:,1]));yaw=np.arctan2(num,den)
RA=(R.from_euler('z',yaw)*R1).as_matrix();tA=om.mean(0)-RA@tp.mean(0);res=np.linalg.norm((RA@tp.T).T+tA-om,axis=1)
# Keep HaWoR's start for the spoon, so the hands stay where they were relative to it at frame 0.
tA=om[0]-RA@tp[0]
TT=np.zeros_like(T);TT[:,:3,:3]=RA@T[:,:3,:3];TT[:,:3,3]=(RA@tp.T).T+tA;TT[:,3,3]=1
W={s:h[f'hand_{s}_wrist_to_world_transform'].astype(float) for s in ('left','right')}
wm=W[a.side][:,:3,3]*scale
dist=np.linalg.norm(wm-TT[:,:3,3],axis=1)
fr=(0,10,20,25,30,34,38,40,45,50,55,60,70,80,100,120,150)
rep=dict(spike_frames=spikes,yaw_deg=float(np.degrees(yaw)),spoon_rise_after_map_cm={f:round(100*float(TT[f,2,3]-TT[0,2,3]),1) for f in (40,50,60)},wrist_rise_metric_cm={f:round(100*float(wm[f,2]-wm[25,2]),1) for f in (40,50,60)},scene_scale=scale,fit_residual_cm=dict(mean=100*float(res.mean()),max=100*float(res.max())),lift_axis_after_map=((TT[50,:3,3]-TT[0,:3,3])/np.linalg.norm(TT[50,:3,3]-TT[0,:3,3])).round(3).tolist(),metric_wrist_to_spoon_cm={f:round(100*float(dist[f]),1) for f in fr},hawor_spoon_move_metric_cm={f:round(100*float(np.linalg.norm(om[f]-om[0])),1) for f in fr},target_spoon_move_cm={f:round(100*float(np.linalg.norm(tp[f]-tp[0])),1) for f in fr},wrist_move_metric_cm_from25={f:round(100*float(np.linalg.norm(wm[f]-wm[25])),1) for f in fr})
rep['hold_note']='';
if a.reanchor:
    s,e=a.hold;rel_p=TT[s,:3,:3].T@(wm[s]-TT[s,:3,3]);new=wm.copy()
    weight=np.clip((np.arange(n)-(s-5))/5.0,0,1)*np.clip(((e+15)-np.arange(n))/15.0,0,1);weight=weight*weight*(3-2*weight)
    anchored=TT[:,:3,3]+np.einsum('tij,j->ti',TT[:,:3,:3],rel_p);new=wm+(anchored-wm)*weight[:,None]
    Rw=R.from_matrix(W[a.side][:,:3,:3]);Ro=R.from_matrix(TT[:,:3,:3]);Rc=R.from_matrix(h['camera_to_world_transform'][:,:3,:3].astype(float));Oc=R.from_rotvec(h[f'hand_{a.side}_wrist_orient_in_camera'].astype(float))
    world=Rc*Oc;held=Ro*Ro[s].inv()*world[s];delta=(world.inv()*held).as_rotvec();final=world*R.from_rotvec(delta*weight[:,None])
    h[f'hand_{a.side}_wrist_orient_in_camera']=(Rc.inv()*final).as_rotvec().astype(np.float32);rep['peak_turn_deg']=round(float(np.degrees(np.linalg.norm(delta*weight[:,None],axis=1).max())),1)
    W2=W[a.side].copy();W2[:,:3,3]=new/scale;h[f'hand_{a.side}_wrist_to_world_transform']=W2.astype(np.float32);rep['reanchor']=dict(hold=[s,e],peak_shift_cm=round(100*float(np.linalg.norm(new-wm,axis=1).max()),1),rel_palm_cm=(100*rel_p).round(1).tolist())
    dist2=np.linalg.norm(new-TT[:,:3,3],axis=1);rep['metric_wrist_to_spoon_after_cm']={f:round(100*float(dist2[f]),1) for f in fr}
O2=TT.copy();O2[:,:3,3]=TT[:,:3,3]/scale;h['object_to_world_transform']=O2.astype(np.float32);h['object_scale']=np.array(1.0/scale,dtype=np.float32);h['gravity_alignment_applied']=np.array(True)
h['object_hawor_metric']=om;a.output.mkdir(parents=True,exist_ok=True);h={k:((np.concatenate([v[:1]]*a.pad+[v],0)[:a.frames] if a.pad else v[:a.frames]) if a.frames and np.ndim(v)>0 and len(v)==n else v) for k,v in h.items()};np.savez(a.output/'result.npz',**h);(a.output/'convert_report.json').write_text(json.dumps(rep,indent=1));print(json.dumps(rep,indent=0))
