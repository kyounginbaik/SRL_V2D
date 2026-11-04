"""Source-only fixed local pot registration, with held-out image/depth audit.

No physical rollout, task metric, checkpoint, or achieved object pose is read.
One SE(3) correction is shared across the complete original trajectory. The
mesh is fixed to the previously source-supported height094 repair. Original
reference files are never edited, and a candidate is exported only if held-out
source evidence improves without losing the initial static registration.
"""
from pathlib import Path
import json,hashlib,time
import cv2,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from repair_episode1_registration import B,EP,OUT,mask,outline

TRAIN=[20,60,100,160,220,280,320]
HELDOUT=[0,5,10,40,80,120,140,180,200,240,260,300,334]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x))+'\n')
def T(x):
 a=np.eye(4);a[:3,:3]=Rotation.from_rotvec(x[3:]).as_matrix();a[:3,3]=x[:3];return a

def main():
    dest=OUT/'source_pose_registration_v1';dest.mkdir(exist_ok=True)
    if (dest/'report.json').exists():raise FileExistsError(dest)
    src=B.parent/'final_recon/episode_000001';cfg=json.loads((src/'config.json').read_text());manifest=json.loads((EP/'input_manifest.json').read_text())
    with np.load(src/'object_poses.npz') as z:poses={k:z[k] for k in z.files}
    names=poses['names'].astype(str).tolist();pi=names.index('white_pot');li=names.index('white_pot_lid')
    with np.load(EP/'camera_trajectory.npz') as z:K=z['K'].copy();w=int(z['width']);h=int(z['height'])
    original=trimesh.load(src/'assets/white_pot_visual.obj',force='mesh',process=False)
    repaired_path=OUT/'pot_geometry_rgb_depth_height094/white_pot_visual.obj';repaired=trimesh.load(repaired_path,force='mesh',process=False)
    samples,_=trimesh.sample.sample_surface(repaired,200000,seed=1801);tree=cKDTree(samples)
    osamples,_=trimesh.sample.sample_surface(original,200000,seed=1801);otree=cKDTree(osamples)
    support=json.loads((B/'refinement_20261018/expanded_batch/source_review/episode_000001/table_support.json').read_text())
    table=np.asarray(support['proposed_setup_from_source']);n=table[2,:3];offset=table[2,3]
    fit=json.loads((OUT/'pot_geometry_rgb_depth_height094/fit.json').read_text());axis=np.asarray(fit['local_height_axis'])
    observations={};data_manifest=[];video=cv2.VideoCapture(manifest['source_video']);frames=sorted(TRAIN+HELDOUT);frame_set=set(frames)
    for frame in range(frames[-1]+1):
        ok,rgb=video.read()
        if not ok:raise ValueError('Source video ended before reference')
        if frame not in frame_set:continue
        dp=Path(cfg['depth'])/f'{frame:06d}.png';raw=cv2.imread(str(dp),-1)
        if raw is None:raise FileNotFoundError(dp)
        dh,dw=raw.shape;Kd=K.copy();Kd[0]*=dw/w;Kd[1]*=dh/h
        mp=Path(cfg['object_masks']['white_pot'])/f'{frame:06d}.png';lp=Path(cfg['object_masks']['white_pot_lid'])/f'{frame:06d}.png'
        target=cv2.imread(str(mp),0)>0;ignore=(cv2.imread(str(lp),0)>0).astype(np.uint8)
        for hand in cfg['hand_tracks']:
            hp=Path(cfg['source'])/'masks'/hand/f'{frame:06d}.png';hm=cv2.imread(str(hp),0)
            if hm is not None:ignore|=(hm>0).astype(np.uint8)
        valid=cv2.dilate(ignore,np.ones((9,9),np.uint8))==0
        observed_full=target&valid;dm=cv2.resize(observed_full.astype(np.uint8),(dw,dh),interpolation=cv2.INTER_NEAREST)>0
        with np.errstate(divide='ignore',invalid='ignore'):depth=65535/raw.astype(float)-1.
        dm&=(depth>.05)&(depth<2.)&np.isfinite(depth)
        yy,xx=np.where(dm);rays=np.c_[xx,yy,np.ones(len(xx))]@np.linalg.inv(Kd).T;cam=rays*depth[yy,xx,None]
        C=poses['world_from_rgb'][frame];world=cam@C[:3,:3].T+C[:3,3]
        if len(world)<250:raise ValueError(f'Insufficient unoccluded source depth at frame{frame}: {len(world)}')
        pick=np.linspace(0,len(world)-1,min(1800,len(world))).astype(int)
        observations[frame]=dict(points=world[pick],rgb=rgb,target=target,valid=valid,K=K,world_from_rgb=C)
        data_manifest.append(dict(frame=frame,role='train' if frame in TRAIN else 'held_out_pose_fit',depth=str(dp),depth_sha256=sha(dp),mask_sha256=sha(mp),lidmask_sha256=sha(lp),points=len(world),sampled_points=len(pick)))
    video.release()
    # All points are expressed in the OLD object frame. Thus one shared C
    # maps the fixed repaired mesh into each unchanged observed trajectory pose.
    local={f:(o['points']-poses['world_from_object'][f,pi,:3,3])@poses['world_from_object'][f,pi,:3,:3] for f,o in observations.items()}
    shift=np.asarray(manifest['setup_from_source'])[:3,3]
    first=np.eye(4);first[:3,:3]=Rotation.from_quat(fit['pose'][3:],scalar_first=True).as_matrix();first[:3,3]=np.asarray(fit['pose'][:3])-shift
    old_to_fit=np.linalg.inv(poses['world_from_object'][0,pi])@first
    starts=[np.zeros(6),np.r_[old_to_fit[:3,3],Rotation.from_matrix(old_to_fit[:3,:3]).as_rotvec()]]
    bounds=(np.r_[[-.05]*3,[-.6]*3],np.r_[[.05]*3,[.6]*3]);fits=[]
    for start_index,x in enumerate(starts):
        history=[];x=np.clip(x,bounds[0]+1e-9,bounds[1]-1e-9)
        for iteration in range(25):
            R=Rotation.from_rotvec(x[3:]).as_matrix();pairs=[]
            for f in TRAIN:
                distance,index=tree.query((local[f]-x[:3])@R)
                keep=distance<min(.04,np.quantile(distance,.88));pairs.append((samples[index[keep]],local[f][keep]))
            def residual(xx):
                C=T(xx);e=[]
                for a,b in pairs:e.append(((a@C[:3,:3].T+C[:3,3]-b)/.003).ravel()/np.sqrt(len(a)))
                # Static frames20 and60 show the pot resting on the measured
                # tabletop. This uses independent source support, not simulation.
                for f in [20,60]:
                    P=poses['world_from_object'][f,pi]@C;bottom=np.min((repaired.vertices@P[:3,:3].T+P[:3,3])@n+offset)
                    e.append(np.array([.15*bottom/.003]));e.append(.15*np.cross(P[:3,:3]@axis,n)/.04)
                e.append(.015*xx/np.r_[[.03]*3,[.35]*3])
                return np.concatenate(e)
            result=least_squares(residual,x,bounds=bounds,loss='soft_l1',f_scale=.15,max_nfev=35)
            change=np.linalg.norm(result.x-x);x=result.x
            row=dict(iteration=iteration,parameters=x.tolist(),cost=float(result.cost));history.append(row)
            if change<1e-6:break
        C=T(x);scores=[]
        for f in TRAIN:
            dist=tree.query((local[f]-C[:3,3])@C[:3,:3])[0];scores.append(float(np.mean(np.minimum(dist,.02))))
        fits.append(dict(parameters=x.tolist(),transform=C.tolist(),training_trimmed_surface_mean_m=float(np.mean(scores)),history=history))
        dump(dest/f'fit_start_{start_index}.json',fits[-1]);print(json.dumps(dict(start=start_index,training_score=fits[-1]['training_trimmed_surface_mean_m'])),flush=True)
    chosen=min(fits,key=lambda x:x['training_trimmed_surface_mean_m']);correction=np.asarray(chosen['transform']);records=[];pictures=[]
    for f in frames:
        o=observations[f];camera={'K':K,'world_from_rgb':o['world_from_rgb'][None]};P=poses['world_from_object'][f,pi];new=P@correction;record=dict(frame=f,time_s=f/20,role='train' if f in TRAIN else 'held_out_pose_fit');panels=[]
        for label,mesh,pose,tr in [('original_mesh_original_pose',original,P,otree),('repaired_mesh_original_pose',repaired,P,tree),('repaired_mesh_source_corrected_pose',repaired,new,tree)]:
            distance=tr.query((o['points']-pose[:3,3])@pose[:3,:3])[0];q=np.r_[pose[:3,3],Rotation.from_matrix(pose[:3,:3]).as_quat(scalar_first=True)]
            prediction=(mask(mesh,q,camera,o['rgb'].shape)>0)&o['valid'];target=o['target']&o['valid']
            record[label]=dict(median_depth_surface_m=float(np.median(distance)),p90_depth_surface_m=float(np.quantile(distance,.9)),visible_mask_iou=float((prediction&target).sum()/max(1,(prediction|target).sum())),bottom_height_above_table_m=float(np.min((mesh.vertices@pose[:3,:3].T+pose[:3,3])@n+offset)))
            panel=o['rgb'].copy();outline(panel,mesh,q,camera,(0,255,0) if 'corrected' in label else (255,170,0));cv2.putText(panel,label.replace('_',' '),(300,675),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
            cv2.putText(panel,f'{f/20:.2f}s {record["role"]}',(300,710),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
            panels.append(cv2.resize(panel[650:1500,300:1800],(600,340)))
        if f in HELDOUT:pictures.append(np.hstack(panels))
        records.append(record)
    held=[r for r in records if r['role']=='held_out_pose_fit'];initial=[r for r in held if r['time_s']<=.5]
    def aggregate(rows,label):return dict(mean_median_depth_mm=float(np.mean([r[label]['median_depth_surface_m']*1000 for r in rows])),mean_p90_depth_mm=float(np.mean([r[label]['p90_depth_surface_m']*1000 for r in rows])),mean_mask_iou=float(np.mean([r[label]['visible_mask_iou'] for r in rows])))
    labels=['original_mesh_original_pose','repaired_mesh_original_pose','repaired_mesh_source_corrected_pose']
    aggregates={scope:{lab:aggregate(rows,lab) for lab in labels} for scope,rows in [('held_out',held),('static_initial',initial)]}
    before=aggregates['held_out']['repaired_mesh_original_pose'];after=aggregates['held_out']['repaired_mesh_source_corrected_pose'];ib=aggregates['static_initial']['repaired_mesh_original_pose'];ia=aggregates['static_initial']['repaired_mesh_source_corrected_pose']
    evidence_pass=after['mean_median_depth_mm']<before['mean_median_depth_mm'] and after['mean_mask_iou']>=before['mean_mask_iou']-.01 and ia['mean_median_depth_mm']<ib['mean_median_depth_mm'] and ia['mean_mask_iou']>=ib['mean_mask_iou']-.01
    report=dict(scope='Independent source-only shared object-local SE3 registration; no physical trajectories or MP scores read',source_object_poses=str(src/'object_poses.npz'),source_object_poses_sha256=sha(src/'object_poses.npz'),fixed_repaired_mesh=str(repaired_path),fixed_repaired_mesh_sha256=sha(repaired_path),train_frames=TRAIN,held_out_pose_frames=HELDOUT,selection='minimum training trimmed surface distance across two initializations; held-out results used only for acceptance',correction_object_local=correction.tolist(),correction_rotation_deg=float(np.degrees(Rotation.from_matrix(correction[:3,:3]).magnitude())),correction_translation_m=correction[:3,3].tolist(),aggregates=aggregates,records=records,data_manifest=data_manifest,source_evidence_pass=bool(evidence_pass),reference_revision_exported=False,limitations=['Shape was previously fitted using seven source frames; these are held-out pose observations, not a wholly held-out shape dataset.','Stereo is reconstructed metric depth, not organizer object ground truth.','Mask IoU excludes dilated hand and other-object segmentation; segmentation errors remain possible.','One fixed local correction cannot repair arbitrary time-varying tracking drift.'])
    cv2.imwrite(str(dest/'heldout_rgb_comparison.jpg'),np.vstack(pictures))
    if evidence_pass:
        candidate={k:v.copy() for k,v in poses.items()};candidate['world_from_object'][:,pi]=candidate['world_from_object'][:,pi]@correction
        assert np.array_equal(candidate['world_from_object'][:,li],poses['world_from_object'][:,li])
        np.savez_compressed(dest/'object_poses_candidate.npz',**candidate);report['reference_revision_exported']=True;report['candidate_sha256']=sha(dest/'object_poses_candidate.npz')
    dump(dest/'report.json',report);print(json.dumps(dict(aggregates=aggregates,source_evidence_pass=evidence_pass,rotation_deg=report['correction_rotation_deg']),default=bool),flush=True)
if __name__=='__main__':main()
