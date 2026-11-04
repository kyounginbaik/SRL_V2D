"""RR2: ball-dominant full-circle roll re-solve. The stored stereo ball detector has almost no detections on fast frames, so 2D ball evidence is re-measured as dark mount blobs (V<60, compact, near the spoon mask, not on hand or cup) in RGB f+1; no ball depth.
Unary = blob-to-model assignment px/30 + WS*sil. Viterbi over circular roll, ends pinned to base, hard rate cap. usage: rr2_solve.py SCAN2 BUNDLE EP PIN_A SOLVE_A SOLVE_B PIN_B OUT [WS=0.3] [CAP=20]"""
import sys, json, numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import linear_sum_assignment as lsa
from scipy.ndimage import gaussian_filter1d as gf
scan,B,EP,PA,SA,SB,PB,OUT=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]),sys.argv[8]
DMAX=float(__import__('os').environ.get('DMAX',14)); WS=float(sys.argv[9]) if len(sys.argv)>9 else .3; CAP=float(sys.argv[10]) if len(sys.argv)>10 else 20.
Cn=f'/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_{EP:06d}'; I=json.load(open(f'{Cn}/intrinsics.json')); fx,fy,cx,cy=I['fx'],I['fy'],I['cx'],I['cy']
z=np.load(scan); R=z['rolls']; fr=z['frames']; S=z['sil']; sh=z['shift']; L,com=z['long'],z['com']; P0=np.load(f'{B}/object_poses.npz',allow_pickle=False); names=[str(x) for x in P0['names']]; k=names.index('wooden_spoon'); Tc=P0['camera_from_object'][:,k]; OTHER=([x for x in names if x!='wooden_spoon'] or [None])[0]; N=len(Tc)
bc=json.load(open('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/out/ballcenters.json'))['per_episode_centres']; BM=np.array([np.median([bc[e][q] for e in bc],0) for q in ('near','far','tip')])
cap=cv2.VideoCapture(f'{Cn}/rgb.mp4'); blobs={}
def Bd(r):
    E=Rot.from_rotvec(L*np.radians(r)).as_matrix(); T=np.eye(4); T[:3,:3]=E; T[:3,3]=com-E@com; return T
for f in fr:
    g=min(int(f)+1,N-1); cap.set(cv2.CAP_PROP_POS_FRAMES,g); ok,im=cap.read(); rd=lambda p:cv2.imread(p,0)>0
    M=rd(f'{Cn}/objects/wooden_spoon/masks/1/{g:06d}.png'); occ=rd(f'{Cn}/objects/wooden_spoon/masks/2/{g:06d}.png')|rd(f'{Cn}/objects/wooden_spoon/masks/3/{g:06d}.png')
    if OTHER: occ|=rd(f'{Cn}/objects/{OTHER}/masks/1/{g:06d}.png')
    band=cv2.dilate(M.astype(np.uint8),np.ones((21,21),np.uint8))>0; dark=(cv2.cvtColor(im,cv2.COLOR_BGR2HSV)[...,2]<60)&band&~cv2.dilate(occ.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    n,lab,st,cen=cv2.connectedComponentsWithStats(dark.astype(np.uint8),8); D=cv2.distanceTransform((~M).astype(np.uint8),cv2.DIST_L2,3); bs=[]
    for i in range(1,n):
        a=st[i,4]; w,h=st[i,2],st[i,3]
        if 12<=a<=500 and max(w,h)<40 and a/(w*h)>.3 and D[int(cen[i,1]),int(cen[i,0])]<DMAX: bs.append(cen[i])
    blobs[int(f)]=np.array(bs).reshape(-1,2)
U=np.zeros_like(S); nb=np.array([len(blobs[int(f)]) for f in fr])
for t,f in enumerate(fr):
    b=blobs[int(f)]
    for i,r in enumerate(R):
        c=0.
        if len(b):
            T=Tc[f]@Bd(r); p=BM@T[:3,:3].T+T[:3,3]; uv=np.c_[fx*p[:,0]/p[:,2]+cx,fy*p[:,1]/p[:,2]+cy]+sh[t,i]; DM=np.minimum(np.linalg.norm(b[:,None]-uv[None],axis=2),30.); a,bb=lsa(DM); c=DM[a,bb].mean()/30.
        U[t,i]=c
Un=U+WS*np.nan_to_num(S,nan=1.); n=len(R); d=np.abs((R[:,None]-R[None]+180)%360-180)
pin=(fr<SA)|(fr>SB); i0=0; Cc=Un[0].copy(); Cc[d[i0]>3]+=100; bp=np.zeros((len(fr),n),int)
for t in range(1,len(fr)):
    P=.01*d/10+100.*(d>CAP); c=Cc[:,None]+P; bp[t]=np.argmin(c,0); Cc=c[bp[t],np.arange(n)]+Un[t]
    if pin[t]: Cc[d[i0]>3]+=100
path=np.zeros(len(fr),int); path[-1]=int(np.argmin(Cc))
for t in range(len(fr)-1,0,-1): path[t-1]=bp[t,path[t]]
ang=np.radians(R[path]); ss,cc=gf(np.sin(ang),1.5,mode='nearest'),gf(np.cos(ang),1.5,mode='nearest'); rs=np.degrees(np.arctan2(ss,cc))
roll=np.zeros(N); shift=np.zeros((N,2)); roll[fr]=rs; shift[fr]=np.c_[gf(sh[np.arange(len(fr)),path,0],1.5,mode='nearest'),gf(sh[np.arange(len(fr)),path,1],1.5,mode='nearest')]; roll[fr[pin]]=0; shift[fr[pin]]=0
np.savez(OUT,roll=roll,tilt=np.zeros((N,2)),shift=shift,frames=fr,raw=R[path],nblob=nb,blobs_u0=U[:,0],blobs_ubest=U.min(1),blob_flat=np.array(json.dumps({int(k):v.tolist() for k,v in blobs.items()})))
w=((R[path]+180)%360)-180; print(f'ep{EP} frames with blobs {int((nb>0).sum())}/{len(fr)}, mean blobs {nb.mean():.1f}; mean |roll| {np.abs(rs).mean():.1f}, max {np.abs(rs).max():.0f}, max step {np.abs(np.diff(np.unwrap(np.radians(rs)))).max()*57.3:.1f}')
print(' ball cost base %.2f -> sol %.2f; per-frame free best |roll|>60 share %.2f'%(U[:,0].mean(),U[np.arange(len(fr)),path].mean(),(np.abs(((R[np.argmin(U,1)]+180)%360)-180)>60).mean()))
print(' f:roll '+' '.join(f'{fr[t]}:{rs[t]:+.0f}' for t in range(0,len(fr),4)))
