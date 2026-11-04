"""Eval episodes: which face (scan twin) does the official registration of our submitted sweeper mesh pick, and which face does the
grey-ball image cue support on the current reconstructed track? Prints per-twin cue scores; twin 0 = the scorer's own choice."""
import sys, json, numpy as np, cv2, trimesh, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0,'/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020/code/sweeper')
import facepick as FP
from scipy.spatial.transform import Rotation as R
PKG='/data1/SRL_V2D/challenge_submissions_SRL/track3_v13_20261022/all_20_episodes'
def submitted_mesh(ep, slot):
    sub=pq.read_table(PKG+'/track3_submission.parquet',columns=['row_id','object_mesh_b64']).to_pandas()
    r=sub[sub.row_id.str.startswith(f'track_3/e{ep:06d}/')&sub.row_id.str.endswith(f'/o{slot}')&(sub.object_mesh_b64.fillna('').str.len()>0)]
    v,f=FP.M._decode_object_mesh(r.object_mesh_b64.iloc[0]); return np.asarray(v,float),np.asarray(f,np.int64)
def scorer_placements(ep, slot):
    cv,cf=submitted_mesh(ep,slot); rv,rf=FP.load(FP.SCAN); seed=1_000_003*ep+101*slot
    cc,cr=FP.M._surface_moments(cv,cf); rc,rr=FP.M._surface_moments(rv,rf); s=rr/cr
    cp=(FP.M._sample_mesh_surface(cv,cf,3000,seed)-cc)*s; rp=FP.M._sample_mesh_surface(rv,rf,3000,seed)-rc
    rot,off=FP.M._align_rigid(rp,cp); c0=cc-rot.T@off/s; base=np.eye(4); base[:3,:3]=rot.T; base[:3,3]=c0-rot.T@rc
    w,U=np.linalg.eigh(np.cov((rv-rv.mean(0)).T)); out=[base]
    for k in range(3):
        S=np.eye(4); S[:3,:3]=R.from_rotvec(np.pi*U[:,k]).as_matrix(); S[:3,3]=rc-S[:3,:3]@rc; out.append(base@S)
    return out, s, (cv,cf)
def run(ep, slot, T, c2w, K, video, mask_dir, step=3):
    pls,s,_=scorer_placements(ep,slot)
    sc=FP.grey_ball_scores(T,pls,video,c2w,K,mask_dir,list(range(0,len(T),step)),idx=(0,1,2,3))
    a,b=sc[0],sc[3]; both=np.isfinite(a)&np.isfinite(b)
    print(f'ep{ep}: scan/ours scale {s:.3f}; grey-ball cue per twin (0 = official registration):',np.round(np.nanmean(sc,1),3),
          f'| frames twin3>twin0 {int((b[both]>a[both]).sum())} / twin0>twin3 {int((a[both]>b[both]).sum())} of {int(both.sum())}')
    return pls, sc

def scorer_placements_mesh(mesh_path, ep, slot):
    """same as scorer_placements but for a mesh file (budgeted exactly like the packer/score_obj)."""
    import sys; sys.path.insert(0,'/data1/SRL_V2D/challenge_submissions_SRL/track3_v13_20261022/all_20_episodes/submission_kit')
    from v2dlb.mesh_budget import budget_mesh
    mv,mf=budget_mesh(mesh_path,4096,4096); cv,cf=np.asarray(mv,float),np.asarray(mf,np.int64)
    rv,rf=FP.load(FP.SCAN); seed=1_000_003*ep+101*slot
    cc,cr=FP.M._surface_moments(cv,cf); rc,rr=FP.M._surface_moments(rv,rf); s=rr/cr
    cp=(FP.M._sample_mesh_surface(cv,cf,3000,seed)-cc)*s; rp=FP.M._sample_mesh_surface(rv,rf,3000,seed)-rc
    rot,off=FP.M._align_rigid(rp,cp); c0=cc-rot.T@off/s; base=np.eye(4); base[:3,:3]=rot.T; base[:3,3]=c0-rot.T@rc
    w,U=np.linalg.eigh(np.cov((rv-rv.mean(0)).T)); out=[base]
    for k in range(3):
        S=np.eye(4); S[:3,:3]=R.from_rotvec(np.pi*U[:,k]).as_matrix(); S[:3,3]=rc-S[:3,:3]@rc; out.append(base@S)
    return out
def relabel(T, pls, k):
    """body-frame relabel so that the official registration (pls[0]) yields the scan face of twin k: T' = T @ pls[k] @ inv(pls[0])."""
    G=pls[k]@np.linalg.inv(pls[0]); return np.einsum('nij,jk->nik',T,G), G
