"""GT-free choice of the sweeper's front/back face: place the scan mesh in our body frame (kit registration of our mesh to the scan,
then each of the scan's 4 principal half-turn twins), pose it with the track, and score each twin against the masked stereo depth
(median point-to-surface distance of back-projected depth points). The twin is a constant body relabel, so the score is summed
over frames. Returns per-twin scores and the 4x4 body_from_scan placements."""
import sys, os, json, math, importlib.util, numpy as np, cv2, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
W='/data1/SRL_V2D/challenge_reconstruction/public_gt_check_20261020'
KIT='/data1/SRL_V2D/challenge_submissions_SRL/track3_v13_20261022/all_20_episodes/submission_kit'
spec=importlib.util.spec_from_file_location('mpsr',KIT+'/metric_code/track_3/MP-SR.py'); M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
SCAN='/data1/SRL_V2D/gt_best_per_object/mini_sweeper.ply'
def load(p): m=trimesh.load(p,force='mesh',process=False); return np.asarray(m.vertices,float),np.asarray(m.faces,np.int64)
def placements(our_mesh, seed=1):
    cv,cf=load(our_mesh); rv,rf=load(SCAN); cc,cr=M._surface_moments(cv,cf); rc,rr=M._surface_moments(rv,rf); s=rr/cr
    cp=(M._sample_mesh_surface(cv,cf,3000,seed)-cc)*s; rp=M._sample_mesh_surface(rv,rf,3000,seed)-rc
    rot,off=M._align_rigid(rp,cp)
    # X_body = c0 + rot^T (X_scan - rc)  (real size), c0 = cc - rot^T off / s
    c0=cc-rot.T@off/s; base=np.eye(4); base[:3,:3]=rot.T; base[:3,3]=c0-rot.T@rc
    w,U=np.linalg.eigh(np.cov((rv-rv.mean(0)).T)); out=[base]
    for k in range(3):
        S=np.eye(4); S[:3,:3]=R.from_rotvec(np.pi*U[:,k]).as_matrix(); S[:3,3]=rc-S[:3,:3]@rc; out.append(base@S)
    return out, s
class Depth:
    def __init__(self, ep, obj, c2w, K, size=(1014,760), mask_dir=None, depth_dir=None):
        self.c2w=c2w; self.K=K; self.size=size; o=f'{W}/outputs/episode_{ep:06d}/{obj}'
        self.md=mask_dir or o+'/masks/1'; self.dd=depth_dir or f'{W}/stereo/episode_{ep:06d}/depth_rgb'
    def points(self, t, erode=3, maxn=1500):
        mk=cv2.imread(f'{self.md}/{t:06d}.png',0); dp=cv2.imread(f'{self.dd}/{t:06d}.png',-1)
        if mk is None or dp is None: return None
        mk=cv2.resize(mk,self.size,interpolation=cv2.INTER_NEAREST)>127
        if erode: mk=cv2.erode(mk.astype(np.uint8),np.ones((2*erode+1,2*erode+1),np.uint8))>0
        d=np.where(dp<65535,65535.0/np.maximum(dp,1)-1,np.nan)
        if d.shape[::-1]!=self.size: d=cv2.resize(d,self.size,interpolation=cv2.INTER_NEAREST)
        ys,xs=np.nonzero(mk&np.isfinite(d)&(d>0.1)&(d<1.5))
        if len(xs)<80: return None
        if len(xs)>maxn: sel=np.random.default_rng(t).choice(len(xs),maxn,replace=False); ys,xs=ys[sel],xs[sel]
        z=d[ys,xs]; P=np.stack([(xs-self.K[0,2])/self.K[0,0]*z,(ys-self.K[1,2])/self.K[1,1]*z,z],1)
        return P@self.c2w[t][:3,:3].T+self.c2w[t][:3,3]
def twin_scores(T, placements_, depth, frames, nsurf=8000, trans_refine=True):
    rv,rf=load(SCAN); surf=trimesh.sample.sample_surface(trimesh.Trimesh(rv,rf),nsurf,seed=0)[0]
    res=np.zeros((len(placements_),0)).tolist(); out=[[] for _ in placements_]
    for t in frames:
        P=depth.points(t)
        if P is None: continue
        for i,Pl in enumerate(placements_):
            X=T[t]@Pl; S=surf@X[:3,:3].T+X[:3,3]; tree=cKDTree(S); d,idx=tree.query(P)
            if trans_refine:  # absorb a small global offset (track position error) before judging the shape fit
                for _ in range(3):
                    sh=np.median(S[idx]-P,0); d,idx=tree.query(P+sh)
            out[i].append(np.median(d)*100)
    return np.array(out)   # placements x frames (cm)

def marker_points(thr=0.004):
    """scan surface points that have no counterpart under the long-axis half turn (the asymmetric bumps = marker balls)."""
    rv,rf=load(SCAN); m=trimesh.Trimesh(rv,rf); pts=trimesh.sample.sample_surface(m,20000,seed=0)[0]; c=rv.mean(0)
    w,U=np.linalg.eigh(np.cov((rv-c).T)); Rk=R.from_rotvec(np.pi*U[:,2]).as_matrix(); q=(pts-c)@Rk.T+c; d,_=cKDTree(pts).query(q)
    return pts[d>thr]
def video_frame(ep, t, cam='a', size=(1014,760)):
    """undistorted stage-A frame (matches K)."""
    o=f'{W}/outputs/episode_{ep:06d}'
    for cand in (f'{o}/mini_sweeper/hawor_native/undistorted', f'{o}/mini_sweeper/frames'):
        pass
    return None

def grey_ball_scores(T, placements_, video, c2w, K, mask_dir, frames, idx=(0,3), radius_px=8, mask_scale=None):
    """Image cue for the front/back face: fraction of projected marker-ball points (scan asymmetric bumps) that land within
    radius_px of grey, low-saturation pixels inside the dilated object mask. Returns placements x frames scores (nan when unusable)."""
    mk3d=marker_points(); cap=cv2.VideoCapture(video); out=np.full((len(placements_),len(frames)),np.nan)
    for fi,t in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES,t); ok,img=cap.read()
        m=cv2.imread(f'{mask_dir}/{t:06d}.png',0)
        if not ok or m is None: continue
        H,Wd=img.shape[:2]; m=cv2.resize(m,(Wd,H),interpolation=cv2.INTER_NEAREST)>127
        m=cv2.dilate(m.astype(np.uint8),np.ones((25,25),np.uint8))>0
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); grey=(hsv[...,1]<45)&(hsv[...,2]>70)&(hsv[...,2]<210)&m
        grey=cv2.morphologyEx(grey.astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
        if grey.sum()<30: continue
        dist=cv2.distanceTransform((1-grey).astype(np.uint8),cv2.DIST_L2,5)
        for i in idx:
            X=np.linalg.inv(c2w[t])@T[t]@placements_[i]; P=mk3d@X[:3,:3].T+X[:3,3]
            if (P[:,2]<=0.05).any(): continue
            uv=P@K.T; uv=(uv[:,:2]/uv[:,2:3]).round().astype(int); ok_=(uv[:,0]>=0)&(uv[:,0]<Wd)&(uv[:,1]>=0)&(uv[:,1]<H)
            if ok_.sum()<20: continue
            out[i,fi]=float((dist[uv[ok_,1],uv[ok_,0]]<radius_px).mean())
    return out

def silhouette_masks(T, our_mesh, c2w, K, size, frames, out_dir, dilate=31):
    """write projected-silhouette masks (dilated) of our mesh for frames lacking SAM masks."""
    import os; os.makedirs(out_dir,exist_ok=True); v,f=load(our_mesh)
    for t in frames:
        X=np.linalg.inv(c2w[t])@T[t]; P=v@X[:3,:3].T+X[:3,3]
        if (P[:,2]<=0.05).any(): continue
        uv=P@K.T; uv=np.round(uv[:,:2]/uv[:,2:3]).astype(np.int32); m=np.zeros(size[::-1],np.uint8)
        cv2.fillPoly(m,[uv[ff] for ff in f],255); m=cv2.dilate(m,np.ones((dilate,dilate),np.uint8)); cv2.imwrite(f'{out_dir}/{t:06d}.png',m)
