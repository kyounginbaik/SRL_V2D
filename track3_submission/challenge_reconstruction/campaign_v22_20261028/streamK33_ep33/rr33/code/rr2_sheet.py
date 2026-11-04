"""before/after with balls: red ring = model ball (near,far) projected, cyan = tip, green dot = dark mount blob; yellow text = occ-aware IoU. usage: rr2_sheet.py EP BUNDLE SCAN SOL OUT frames"""
import sys, json, numpy as np, cv2, trimesh
sys.path.insert(0,'/data1/SRL_V2D/challenge_reconstruction/campaign_v22_20261028/streamK33_ep33/rr33/code'); from rr_apply import apply
EP=int(sys.argv[1]);Bn=sys.argv[2];scan=np.load(sys.argv[3]);sol=np.load(sys.argv[4]);O=sys.argv[5];fl=[int(x) for x in sys.argv[6].split(',')]
n=f'episode_{EP:06d}';E=f'/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/{n}';z=np.load(f'{Bn}/object_poses.npz');C1,_,k=apply(z,scan,sol);C0=z['camera_from_object'][:,k];names=[str(x) for x in z['names']];OTHER=([x for x in names if x!='wooden_spoon'] or [None])[0]
blobs=json.loads(str(sol['blob_flat']))
g=trimesh.load('/data1/SRL_V2D/challenge_submissions_SRL/track3_v30c1_cwmesh_nogt_20261101/all_20_episodes/meshes/episode_000033/wooden_spoon.obj',force='mesh',process=False);V=np.asarray(g.vertices);F=np.asarray(g.faces)
bc=json.load(open('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/streamG18_spoonmesh/out/ballcenters.json'))['per_episode_centres'];BM=np.array([np.median([bc[e][q] for e in bc],0) for q in ('near','far','tip')])
I=json.load(open(f'{E}/intrinsics.json'));fx,fy,cx,cy=I['fx'],I['fy'],I['cx'],I['cy'];cap=cv2.VideoCapture(f'{E}/rgb.mp4')
pr=lambda T,P:(lambda p:np.c_[p[:,0]/p[:,2]*fx+cx,p[:,1]/p[:,2]*fy+cy])(P@T[:3,:3].T+T[:3,3])
tiles=[]
for f in fl:
    gi=f+1;rd=lambda p:cv2.imread(p,0)>0;M=rd(f'{E}/objects/wooden_spoon/masks/1/{gi:06d}.png');occ=rd(f'{E}/objects/wooden_spoon/masks/2/{gi:06d}.png')|rd(f'{E}/objects/wooden_spoon/masks/3/{gi:06d}.png')
    if OTHER: occ|=rd(f'{E}/objects/{OTHER}/masks/1/{gi:06d}.png')
    cap.set(cv2.CAP_PROP_POS_FRAMES,gi);ok,im=cap.read();t=[];ys,xs=np.where(M);x0=int(np.clip(xs.mean()-170,0,im.shape[1]-340));y0=int(np.clip(ys.mean()-170,0,im.shape[0]-340))
    for T,lab in ((C0[f],'before'),(C1[f],'after')):
        S=np.zeros(M.shape,np.uint8);uv=np.round(pr(T,V)).astype(np.int32)
        for tt in uv[F]: cv2.fillConvexPoly(S,tt,1)
        S=S>0;q=(S&M).sum()/max((M|(S&~M&~occ)).sum(),1);ov=im.copy();c,_=cv2.findContours(S.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);cv2.drawContours(ov,c,-1,(0,0,255),1)
        for p,col in zip(pr(T,BM),((0,0,255),(255,0,255),(255,255,0))): cv2.circle(ov,(int(p[0]),int(p[1])),9,col,2)
        for b in blobs.get(str(f),[]): cv2.circle(ov,(int(b[0]),int(b[1])),3,(0,255,0),-1)
        cc=ov[y0:y0+340,x0:x0+340].copy();cv2.putText(cc,f'f{f} {lab} {q:.2f}',(4,22),0,.6,(0,255,255),2);t.append(cc)
    tiles.append(np.hstack(t))
w=[np.hstack(tiles[i:i+3]+[np.zeros_like(tiles[0])]*(3-len(tiles[i:i+3]))) for i in range(0,len(tiles),3)];cv2.imwrite(O+'.jpg',np.vstack(w),[cv2.IMWRITE_JPEG_QUALITY,78])
