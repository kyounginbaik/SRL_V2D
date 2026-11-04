"""before/after overlay sheet (pose f on RGB f+1); strict silhouette per second with occlusion. usage: rr_sheet.py EP SCAN SOL OUTPREFIX frames(start:stop:step)"""
import sys, json, numpy as np, cv2, trimesh
sys.path.insert(0,'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/rr33/code'); from rr_apply import apply
EP=int(sys.argv[1]); scan=np.load(sys.argv[2]); sol=np.load(sys.argv[3]); O=sys.argv[4]; a,b,st=[int(x) for x in sys.argv[5].split(':')]
n=f'episode_{EP:06d}'; E=f'/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/{n}'; z=np.load(f'/data1/SRL_V2D/challenge_submissions_SRL/best_reconstruction_20261103/bundles/{n}/object_poses.npz')
C1,_,k=apply(z,scan,sol); C0=z['camera_from_object'][:,k]; names=[str(x) for x in z['names']]; OTHER=([x for x in names if x!='wooden_spoon'] or [None])[0]
g=trimesh.load('/data1/SRL_V2D/challenge_submissions_SRL/track3_v30c1_cwmesh_nogt_20261101/all_20_episodes/meshes/episode_000033/wooden_spoon.obj',force='mesh',process=False); V=np.asarray(g.vertices);F=np.asarray(g.faces)
I=json.load(open(f'{E}/intrinsics.json')); fx,fy,cx,cy=I['fx'],I['fy'],I['cx'],I['cy']; cap=cv2.VideoCapture(f'{E}/rgb.mp4'); N=len(C0)
def rast(T,sh):
    p=V@T[:3,:3].T+T[:3,3]; uv=np.round(np.c_[p[:,0]/p[:,2]*fx+cx,p[:,1]/p[:,2]*fy+cy]).astype(np.int32); S=np.zeros(sh,np.uint8)
    for t in uv[F]: cv2.fillConvexPoly(S,t,1)
    return S>0
res=[];tiles=[]
for f in range(N-1):
    g_=f+1; rd=lambda p: cv2.imread(p,0)>0
    M=rd(f'{E}/objects/wooden_spoon/masks/1/{g_:06d}.png'); occ=rd(f'{E}/objects/wooden_spoon/masks/2/{g_:06d}.png')|rd(f'{E}/objects/wooden_spoon/masks/3/{g_:06d}.png')
    if OTHER: occ|=rd(f'{E}/objects/{OTHER}/masks/1/{g_:06d}.png')
    S0,S1=rast(C0[f],M.shape),rast(C1[f],M.shape); bg=~M&~occ
    q=lambda S:(int((S&M).sum())/max(int((M|(S&bg)).sum()),1))
    res.append((q(S0),q(S1)))
    if a<=f<b and (f-a)%st==0:
        cap.set(cv2.CAP_PROP_POS_FRAMES,g_); ok,im=cap.read(); t=[]
        ys,xs=np.where(M|S0); x0=int(np.clip(xs.mean()-170,0,im.shape[1]-340)); y0=int(np.clip(ys.mean()-170,0,im.shape[0]-340))
        cs,_=cv2.findContours(M.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for S,lab in ((S0,'before'),(S1,'after')):
            ov=im.copy(); ct,_=cv2.findContours(S.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(ov,ct,-1,(0,0,255),2); c=ov[y0:y0+340,x0:x0+340].copy(); cv2.putText(c,f'f{f} {lab} {q(S):.2f}',(4,22),0,.6,(0,255,255),2); t.append(c)
        tiles.append(np.hstack(t))
res=np.array(res); np.save(O+'.npy',res)
print(f'ep{EP} occ-aware IoU mean before {res[:,0].mean():.3f} after {res[:,1].mean():.3f}'); print(' per s: '+' '.join(f'{s//20}s:{res[s:s+20,0].mean():.2f}>{res[s:s+20,1].mean():.2f}' for s in range(0,len(res),20)))
if tiles:
    w=[np.hstack(tiles[i:i+3]+[np.zeros_like(tiles[0])]*(3-len(tiles[i:i+3]))) for i in range(0,len(tiles),3)]; cv2.imwrite(O+'.jpg',np.vstack(w),[cv2.IMWRITE_JPEG_QUALITY,75])
