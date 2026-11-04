"""RR33 per-frame scan (GT-free, no public data). For pose f vs RGB frame f+1 (known RGB lag), for each body roll about the long axis through COM (-90..+90, 3 deg) find
the best small mid/short tilt (+-8) and image shift (+-1.5 cm) under an occlusion-aware silhouette cost: (FN + FP on known background)/|mask|. Hand masks, cup mask and pixels
whose stereo depth is in front of the base render are unknown, not background. Secondary: 2D ball reprojection, stereo depth residual (pose f vs depth f). Half-res raster.
usage: rr_scan.py BUNDLE_EP_DIR EP OUT.npz"""
import sys, os, json, numpy as np, cv2, trimesh
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
cv2.setNumThreads(1)
from multiprocessing import Pool
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import linear_sum_assignment
B, EP, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
C = f'/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_{EP:06d}'
K = json.load(open(f'{C}/intrinsics.json')); fx, fy, cx, cy, W, H = K['fx'], K['fy'], K['cx'], K['cy'], K['width'], K['height']
z = np.load(f'{B}/object_poses.npz', allow_pickle=False); names = [str(n) for n in z['names']]; KS = names.index('wooden_spoon'); Tc = z['camera_from_object'][:, KS]; Wr = z['world_from_rgb']; N = len(Tc)
OTHER = ([n for n in names if n != 'wooden_spoon'] or [None])[0]
m0 = trimesh.load(f'{B}/meshes/wooden_spoon.obj', force='mesh'); v0 = np.asarray(m0.vertices); com = np.asarray(m0.center_mass)
c0 = v0 - v0.mean(0); _, U = np.linalg.eigh(c0.T @ c0 / len(c0)); LONG, MID, SHORT = U[:, 2].copy(), U[:, 1].copy(), U[:, 0].copy()
for a in (LONG, MID, SHORT):
    if a[np.argmax(np.abs(a))] < 0: a *= -1
G18 = '/data1/SRL_V2D/challenge_submissions_SRL/track3_v30c1_cwmesh_nogt_20261101/all_20_episodes/meshes/episode_000033/wooden_spoon.obj'   # our GT-free G18 mesh, same body frame, with balls
g = trimesh.load(G18, force='mesh', process=False); v = np.asarray(g.vertices); f_ = np.asarray(g.faces)
bc = json.load(open('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/out/ballcenters.json'))['per_episode_centres']
BM = np.array([np.median([bc[e][k] for e in bc], 0) for k in ('near', 'far', 'tip')])
BD = json.load(open(f'{C}/ball_detections_wooden_spoon.json'))['frames']
ROLLS = np.arange(0, 360, 3.0); TILT = [0.]; FRA, FRB = [int(x) for x in os.environ['FR'].split(':')]
def rot(r, m, s): return Rot.from_rotvec(LONG*np.radians(r)).as_matrix() @ Rot.from_rotvec(MID*np.radians(m)).as_matrix() @ Rot.from_rotvec(SHORT*np.radians(s)).as_matrix()
def body(E):
    T = np.eye(4); T[:3,:3] = E; T[:3,3] = com - E @ com; return T
def rd(p):
    x = cv2.imread(p, cv2.IMREAD_UNCHANGED); return x
def project(T, pts):
    p = pts @ T[:3,:3].T + T[:3,3]; return np.c_[fx*p[:,0]/p[:,2]+cx, fy*p[:,1]/p[:,2]+cy], p[:,2]
def raster(T, x0, y0, w, h, zbuf=False):
    uv, zz = project(T, v); uv = np.round((uv - [x0, y0]) * 0.5).astype(np.int32); tri = uv[f_]
    if zbuf:
        zb = np.full((h, w), np.inf, np.float32); fz = zz[f_].mean(1)
        for t in np.argsort(-fz): cv2.fillConvexPoly(zb, tri[t], float(fz[t]))
        return zb
    S = np.zeros((h, w), np.uint8)
    for t in tri: cv2.fillConvexPoly(S, t, 1)
    return S
def one(t):
    g = min(t+1, N-1)
    M = rd(f'{C}/objects/wooden_spoon/masks/1/{g:06d}.png') > 0
    occ = (rd(f'{C}/objects/wooden_spoon/masks/2/{g:06d}.png') > 0) | (rd(f'{C}/objects/wooden_spoon/masks/3/{g:06d}.png') > 0)
    if OTHER: occ |= rd(f'{C}/objects/{OTHER}/masks/1/{g:06d}.png') > 0
    occ = cv2.dilate(occ.astype(np.uint8), np.ones((5,5),np.uint8)) > 0
    dv = rd(f'{C}/depth/{t:06d}.png').astype(np.float64); D = np.where(dv < 65535, 65535/np.maximum(dv,1)-1, np.nan)
    n = len(ROLLS); nball = np.zeros(n); sil = np.full(n, np.nan); ball = np.full(n, np.nan); dep = np.full(n, np.nan); tilt = np.zeros((n,2)); shift = np.zeros((n,2))
    uv0,_ = project(Tc[t], v); mpx = int(M.sum()); xs = np.r_[uv0[:,0], np.nonzero(M)[1]]; ys = np.r_[uv0[:,1], np.nonzero(M)[0]]
    x0,y0 = int(max(np.percentile(xs,0)-60,0)), int(max(np.percentile(ys,0)-60,0)); x1,y1 = int(min(xs.max()+60,W)), int(min(ys.max()+60,H))
    w,h = (x1-x0)//2+1, (y1-y0)//2+1
    # half-res data
    rs = lambda a: cv2.resize(a.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST) > 0
    pad = np.zeros((H+2,W+2),bool)
    Mh = rs(M[y0:y1,x0:x1]); Oh = rs(occ[y0:y1,x0:x1]); Dh = cv2.resize(np.nan_to_num(D,nan=99.)[y0:y1,x0:x1].astype(np.float32),(w,h),interpolation=cv2.INTER_NEAREST)
    zb0 = raster(Tc[t], x0,y0,w,h,zbuf=True); zb0f = np.where(np.isfinite(zb0), zb0, 99.); zmin = cv2.erode(zb0f.astype(np.float32), np.ones((21,21),np.uint8))
    Oh = Oh | ((zmin < 90) & (Dh < zmin - 0.02))
    Bg = ~Mh & ~Oh; A = np.fft.rfft2(Mh.astype(np.float32) - Bg.astype(np.float32))
    Mc = max(Mh.sum(), 1); zm = float(np.median(project(Tc[t], v)[1])); WIN = int(round(0.015*fx/zm*0.5))
    def ev(Tb):
        S = raster(Tb, x0,y0,w,h); cc = np.fft.irfft2(A * np.conj(np.fft.rfft2(S.astype(np.float32))), s=(h,w))    # corr(M-B, S shifted by d)
        best=None
        idx=np.arange(-WIN,WIN+1); sub=cc[np.ix_(idx%h, idx%w)]
        j=np.unravel_index(np.argmax(sub),sub.shape); return (Mc-float(sub[j]))/Mc, int(idx[j[1]])*2, int(idx[j[0]])*2, S
    base = (Mh.sum() - 0, )
    S00 = raster(Tc[t], x0,y0,w,h); sil00 = ((Mh & ~S00.astype(bool)).sum() + (S00.astype(bool) & Bg).sum())/Mc
    if mpx >= 300:
      for i,r in enumerate(ROLLS):
        best = None; mb, sb = 0., 0.
        for mm in TILT:
            c,du,dv_,_ = ev(Tc[t] @ body(rot(r,mm,0.))); c += 0.002*abs(mm)
            if best is None or c < best[0]: best=(c,du,dv_,mm,0.)
        mb = best[3]
        for ss in TILT:
            if ss == 0: continue
            c,du,dv_,_ = ev(Tc[t] @ body(rot(r,mb,ss))); c += 0.002*(abs(mb)+abs(ss))
            if c < best[0]: best=(c,du,dv_,mb,ss)
        sil[i],shift[i,0],shift[i,1],tilt[i,0],tilt[i,1] = best[0],best[1],best[2],best[3],best[4]
        Tb = Tc[t] @ body(rot(r,tilt[i,0],tilt[i,1]))
        det = BD[g] or []
        if len(det):
            P,_ = project(Tb, BM); P = P + shift[i]; uvd = np.array([d['uv'] for d in det]); DM = np.minimum(np.linalg.norm(uvd[:,None]-P[None],axis=2),40.); ri,ci = linear_sum_assignment(DM); ball[i] = float(DM[ri,ci].mean()); nball[i] = len(det)
        zb = raster(Tb, x0,y0,w,h,zbuf=True); zs = np.roll(np.roll(zb, int(shift[i,1]/2), 0), int(shift[i,0]/2), 1)
        vis = np.isfinite(zs) & Mh & ~Oh & (Dh < 90)
        if vis.sum() > 40:
            d_ = zs[vis]-Dh[vis]; md = np.median(d_); dep[i] = 1000*float(np.mean(np.minimum(np.abs(d_-md),0.02)))
    Rw = np.stack([Wr[t][:3,:3] @ (Tc[t] @ body(rot(r,tilt[i,0],tilt[i,1])))[:3,:3] for i,r in enumerate(ROLLS)])
    return t, mpx, sil00, sil, ball, dep, tilt, shift, Rw, zm, nball.max()
if __name__ == '__main__':
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as P: rows = P.map(one, range(FRA, min(FRB,N)), chunksize=1)
    rows.sort(key=lambda r: r[0])
    np.savez_compressed(OUT, frames=np.array([r[0] for r in rows]), nball=np.array([r[10] for r in rows]), rolls=ROLLS, maskpx=np.array([r[1] for r in rows]), sil00=np.array([r[2] for r in rows]), sil=np.stack([r[3] for r in rows]), ball=np.stack([r[4] for r in rows]), dep=np.stack([r[5] for r in rows]),
        tilt=np.stack([r[6] for r in rows]), shift=np.stack([r[7] for r in rows]), Rw=np.stack([r[8] for r in rows]).astype(np.float32), zm=np.array([r[9] for r in rows]), long=LONG, mid=MID, short=SHORT, com=com, bundle=B)
    print('done', N)
