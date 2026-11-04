import sys,subprocess,numpy as np,cv2
sys.argv=['r','x','balls=1'];exec(open('refine.py').read().split("if __name__")[0])
SC=1.0;Tn=np.load('full_0.1.npz')['T'];Tcn=np.load('cupfull_0.1.npz')['T'];kc=list(Z['names']).index('blue_cup');Tc=Z['camera_from_object'][:,kc]
mc=trimesh.load(f'{KD}/meshes/blue_cup.obj',force='mesh')
def mk(mm):return np.asarray(mm.vertices),np.asarray(mm.faces),np.asarray(mm.face_normals)
MS=mk(m);MC=mk(mc)
def draw(I,T,M,col):
    v_,f_,n_=M;p=v_@T[:3,:3].T+T[:3,3];u=proj(p).round().astype(np.int32);fz=p[f_,2].mean(1);nc=n_@T[:3,:3].T;cc=p[f_].mean(1);vw=-cc/np.linalg.norm(cc,axis=1,keepdims=True);a=np.zeros_like(I);ma=np.zeros(I.shape[:2],np.uint8)
    for t in np.argsort(-fz):
        s=(nc[t]*vw[t]).sum()
        if s>0:cv2.fillConvexPoly(a,u[f_[t]],tuple(int(c*(.45+.55*s)) for c in col));cv2.fillConvexPoly(ma,u[f_[t]],1)
    I[ma>0]=(.3*I[ma>0]+.7*a[ma>0]).astype(np.uint8)
def lab(I,t):
    I=cv2.resize(I,(640,480));cv2.rectangle(I,(0,0),(640,34),(20,20,20),-1);cv2.putText(I,t,(8,24),0,.65,(255,255,255),2);return I
cap=cv2.VideoCapture(f'{C}/rgb.mp4');out='ep32_recon_old_vs_new_v2.mp4';wr=None;N=len(Tn)
ff=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x480','-r','20','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-crf','20',out],stdin=subprocess.PIPE)
i=0
while True:
    ok,im=cap.read()
    if not ok:break
    p=min(max(i-1,0),N-1);A=im.copy();B=im.copy();D=im.copy()
    for X,Ts,Tq,col in ((B,T0,Tc,(70,70,255)),(D,Tn,Tcn,(70,220,70))):
        draw(X,Tq[p],MC,(220,150,60));draw(X,Ts[p],MS,col)
    cv2.putText(A,f'{i/20:.2f}s',(10,110),0,1.4,(255,255,255),3)
    ff.stdin.write(np.hstack([lab(A,'Original video'),lab(B,'OLD spoon + cup'),lab(D,'NEW spoon + cup')]).tobytes());i+=1
ff.stdin.close();ff.wait();print(i)
