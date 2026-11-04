import sys,subprocess,json,numpy as np,cv2,trimesh
C='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000032'
K=json.load(open(f'{C}/intrinsics.json'));fx,fy,cx,cy,W,H=K['fx'],K['fy'],K['cx'],K['cy'],K['width'],K['height']
KD='../bundles/K32_R10/episode_000032/meshes/'
def mk(p):m=trimesh.load(p,force='mesh');return np.asarray(m.vertices),np.asarray(m.faces),np.asarray(m.face_normals)
MS=mk(KD+'wooden_spoon.obj');MC=mk(KD+'blue_cup.obj')
d=np.load('cmp_tracks.npz');tracks={'old':(d['T0'],d['Oc']),'final':(np.load('mergedw_spoon.npy'),np.load('cup_lam0.4_sg_rot9_pos7.npy'))}
def proj(p):return np.c_[fx*p[:,0]/p[:,2]+cx,fy*p[:,1]/p[:,2]+cy]
def render(im,objs):
    """objs: list of (T,(v,f,n),color). Joint painter's algorithm across objects by face depth."""
    tris=[];cols=[];dep=[]
    for T,(v,f,n),col in objs:
        p=v@T[:3,:3].T+T[:3,3];u=proj(p);nc=n@T[:3,:3].T;cc=p[f].mean(1);vw=-cc/np.linalg.norm(cc,axis=1,keepdims=True);s=(nc*vw).sum(1);keep=s>0
        tris.append(u[f[keep]]);dep.append(cc[keep,2]);cols.append(np.array([[int(c*(.45+.55*x)) for c in col] for x in s[keep]]))
    tris=np.concatenate(tris);dep=np.concatenate(dep);cols=np.concatenate(cols);layer=np.zeros_like(im);mask=np.zeros(im.shape[:2],np.uint8)
    for t in np.argsort(-dep):
        pts=tris[t].round().astype(np.int32);cv2.fillConvexPoly(layer,pts,tuple(int(c) for c in cols[t]));cv2.fillConvexPoly(mask,pts,1)
    m=mask>0;im[m]=(.4*im[m]+.6*layer[m]).astype(np.uint8);return im
def lab(I,t):
    I=cv2.resize(I,(800,600),interpolation=cv2.INTER_AREA);cv2.rectangle(I,(0,0),(800,34),(20,20,20),-1);cv2.putText(I,t,(8,24),0,.65,(255,255,255),2,cv2.LINE_AA);return I
cap=cv2.VideoCapture(f'{C}/rgb.mp4');out='ep32_recon_old_vs_final_RP2_depth.mp4';N=len(d['T0'])
ff=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','2400x600','-r','20','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-crf','20',out],stdin=subprocess.PIPE)
i=0
while True:
    ok,im=cap.read()
    if not ok:break
    p=min(max(i-1,0),N-1);A=im.copy();X=im.copy();Y=im.copy()
    render(X,[(tracks['old'][1][p],MC,(220,150,60)),(tracks['old'][0][p],MS,(60,60,255))]);render(Y,[(tracks['final'][1][p],MC,(220,150,60)),(tracks['final'][0][p],MS,(70,220,70))])
    cv2.putText(A,f'{i/20:.2f}s',(10,110),0,1.4,(255,255,255),3)
    ff.stdin.write(np.hstack([lab(A,'Original'),lab(X,'OLD K32_R10 (depth-ordered)'),lab(Y,'FINAL K32_RP2 (depth-ordered)')]).tobytes());i+=1
ff.stdin.close();ff.wait();print(i)
