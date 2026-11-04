"""Read-only multi-frame source test of cup32 axial height; no adopted repair."""
from pathlib import Path
import json
import cv2,numpy as np,trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from repair_episode1_registration import mask,outline
B=Path(__file__).resolve().parents[1]

def main():
    source=B.parent/'final_recon/episode_000032';inp=B/'episodes/episode_000032';out=B/'refinement_20261018/expanded_batch/source_review/episode_000032/height_audit_v2';out.mkdir(parents=True,exist_ok=True)
    config=json.loads((source/'config.json').read_text());manifest=json.loads((inp/'input_manifest.json').read_text());A=np.array(manifest['setup_from_source']);poses=np.load(source/'object_poses.npz');slot=list(poses['names']).index('blue_cup')
    geom=json.loads((inp/'metric_geometry.json').read_text());axis=np.array(geom['objects']['blue_cup']['axis_local']);axis/=np.linalg.norm(axis)
    # Symmetry axes are unsigned. Select the upward direction so the preserved
    # rim is the actual upper rim, rather than the bottom used in the v1 scan.
    initial=A@poses['world_from_object'][0,slot]
    if (initial[:3,:3]@axis)[2]<0:axis=-axis
    original=trimesh.load(source/'assets/blue_cup_visual.obj',process=False);top=float(np.max(original.vertices@axis));samples,_=trimesh.sample.sample_surface(original,90000,seed=32)
    camera=dict(np.load(inp/'camera_trajectory.npz'));clouds=out.parent/'stereo_clouds';rows=[];panels=[]
    for scale in [1.,.975,.95,.925,.90,.875,.85]:
        mesh=original.copy();mesh.vertices+=np.outer((scale-1)*(mesh.vertices@axis-top),axis);s=samples+np.outer((scale-1)*(samples@axis-top),axis)
        frame_reports=[]
        for frame in [0,5,10,20,40,60]:
            z=np.load(clouds/f'blue_cup_{frame:06d}.npz');T=z['world_from_object'];points=(z['points_source']-T[:3,3])@T[:3,:3]
            dist=cKDTree(s).query(points)[0];rgb=cv2.imread(str(clouds/f'rgb_{frame:06d}.jpg'));H,W=rgb.shape[:2]
            observed=cv2.imread(str(Path(config['object_masks']['blue_cup'])/f'{frame:06d}.png'),0);observed=cv2.resize(observed,(W,H),interpolation=cv2.INTER_NEAREST)>0;hand=np.zeros((H,W),np.uint8)
            for side in config['hand_tracks']:
                h=cv2.imread(str(Path(config['source'])/'masks'/side/f'{frame:06d}.png'),0)
                if h is not None:hand|=cv2.resize(h,(W,H),interpolation=cv2.INTER_NEAREST)
            keep=~(cv2.dilate(hand,np.ones((7,7),np.uint8))>0);observed&=keep
            C=dict(camera,world_from_rgb=camera['world_from_rgb'][frame:frame+1]);P=A@poses['world_from_object'][frame,slot];pose=np.r_[P[:3,3],Rotation.from_matrix(P[:3,:3]).as_quat(scalar_first=True)]
            projected=(mask(mesh,pose,C,rgb.shape)>0)&keep
            frame_reports.append(dict(frame=frame,surface_p50_m=float(np.median(dist)),surface_p80_m=float(np.quantile(dist,.8)),visible_mask_iou=float((projected&observed).sum()/max(1,(projected|observed).sum()))))
            if scale in [1.,.9] and frame in [0,40,60]:
                panel=rgb.copy();outline(panel,mesh,pose,C,(0,0,255));ys,xs=np.where(observed);crop=panel[max(0,ys.min()-60):min(H,ys.max()+60),max(0,xs.min()-60):min(W,xs.max()+60)];crop=cv2.resize(crop,(500,500));cv2.putText(crop,f'frame {frame}; height {scale:.2f}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2);panels.append(crop)
        initial=A@poses['world_from_object'][0,slot];bottom=float(np.min((mesh.vertices@initial[:3,:3].T+initial[:3,3])[:,2]))
        rows.append(dict(axial_scale=scale,initial_visual_bottom_m=bottom,frames=frame_reports))
    cv2.imwrite(str(out/'height_comparison.jpg'),np.vstack([np.hstack(panels[:3]),np.hstack(panels[3:])]))
    report=dict(episode='episode_000032',source_files_unchanged=True,original_poses_unchanged=True,adopted_repair=False,
        parameterization='Axial mesh scaling about the upward-facing original top rim; this corrects the signed-axis ambiguity in height_audit. Source-evidence evaluation only, not a physical or metric improvement claim',axis_local_upward=axis.tolist(),scan=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(scale=r['axial_scale'],bottom_m=r['initial_visual_bottom_m'],mean_iou=float(np.mean([f['visible_mask_iou'] for f in r['frames']])),mean_depth_m=float(np.mean([f['surface_p50_m'] for f in r['frames']]))) for r in rows]),flush=True)

if __name__=='__main__':main()
