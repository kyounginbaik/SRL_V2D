"""Per-frame check of flagged spoon frames: tracked pose vs pose rolled 180 deg about the handle line (IoU half-res + ball px)."""
import sys, json, numpy as np, trimesh
sys.path.insert(0,'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamZ9_anchor_spoon/code')
from z9_anchor import frame_inputs, load_k, ball_px, evidence
from pathlib import Path
R=Path('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026')
out={}
for ep, frames in [(3,[124,125,126,127,128]),(32,[89,90,91,121,122,123,124,125,149,150,151,152,153,154,155]),(37,[49,50,51,52,53]),(33,[157,158,159,189,190,191,201,202,203]),(36,[83,84,85,86,87])]:
    root=R/f'canonical/evaluation/episode_{ep:06d}'; fol=R/f'selected20_final_v5/episode_{ep:06d}'
    d=np.load(fol/'object_poses.npz'); names=[str(n) for n in d['names']]; s=names.index('wooden_spoon'); cam=d['camera_from_object'][:,s]
    mesh=trimesh.load(fol/'meshes/wooden_spoon.obj',force='mesh',process=False)
    pts,fc=trimesh.sample.sample_surface(mesh,20000,seed=260126); nrm=np.asarray(mesh.face_normals)[fc]
    hp=pts[pts[:,0]<-.05]; yc,zc=np.median(hp[:,1]),np.median(hp[:,2])
    F=np.eye(4); F[:3,:3]=np.diag([1,-1,-1.]); F[:3,3]=[0,2*yc,2*zc]
    k,intr=load_k(root,507,380); kf=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
    model=np.array(json.loads((root/'exports/spoon_roll_v8/selection.json').read_text())['objects']['wooden_spoon']['ball_roll']['model_body_m'])
    dets=json.loads((root/'ball_detections_wooden_spoon.json').read_text())['frames']
    for t in frames:
        fi=frame_inputs(root,'wooden_spoon',names,t,507,380)
        if fi is None: print(ep,t,'nomask'); continue
        m,ex,dp,hpx=fi
        r=[]
        for P in [cam[t], cam[t]@F]:
            _,info=evidence(mesh,pts,nrm,P,k,m,ex,dp); r.append((info['visible_iou'], ball_px(model,dets[t] if t<len(dets) else [],P,kf), info['median_z_mm']))
        print(ep,t,'mask',int(m.sum()),'hand',hpx,'nb',len(dets[t]) if t<len(dets) else 0,'pose iou %.3f ball %.1f z %s | flip iou %.3f ball %.1f z %s'%(r[0][0],r[0][1],r[0][2] and round(r[0][2],1),r[1][0],r[1][1],r[1][2] and round(r[1][2],1)),flush=True)
