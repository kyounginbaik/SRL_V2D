import sys,subprocess,numpy as np,cv2
sys.argv=['r','x','balls=1'];exec(open('refine.py').read().split("if __name__")[0])
SC=1.0;d=dict(np.load('cmp_tracks.npz'));ms=np.load('mgw_spoon_0.npz');mcup=np.load('mg_cup_0.npz');d['Ms']=ms['T'];d['Mc']=mcup['T'];d['Mc2']=np.load('merged_cup_tracks.npy');mc2=np.load('mc2_0.npz');Sp2=mc2['B'];Sp={'wooden_spoon_merged':ms['B'],'blue_cup_merged':mcup['B']};_o=np.load('spikes.npz');Sp.update({k:_o[k] for k in _o.files});mc=trimesh.load(f'{KD}/meshes/blue_cup.obj',force='mesh')
DC=json.load(open(f'{C}/ball_detections_blue_cup.json'))['frames']
def mk(mm):return np.asarray(mm.vertices),np.asarray(mm.faces),np.asarray(mm.face_normals)
MS=mk(m);MC=mk(mc)
def draw(I,T,M,col):
    v_,f_,n_=M;p=v_@T[:3,:3].T+T[:3,3];u=proj(p).round().astype(np.int32);fz=p[f_,2].mean(1);nc=n_@T[:3,:3].T;cc=p[f_].mean(1);vw=-cc/np.linalg.norm(cc,axis=1,keepdims=True);a=np.zeros_like(I);ma=np.zeros(I.shape[:2],np.uint8)
    for t in np.argsort(-fz):
        s=(nc[t]*vw[t]).sum()
        if s>0:cv2.fillConvexPoly(a,u[f_[t]],tuple(int(c*(.45+.55*s)) for c in col));cv2.fillConvexPoly(ma,u[f_[t]],1)
    I[ma>0]=(.45*I[ma>0]+.55*a[ma>0]).astype(np.uint8)
def spikes(I,T,S,col):
    for b in proj(S@T[:3,:3].T+T[:3,3]):cv2.circle(I,tuple(np.int32(b)),9,col,2,cv2.LINE_AA)
def dets(I,p):
    for D_ in (DET,DC):
        for x in (D_[p+1] if p+1<len(D_) else []):
            if x.get('depth_mad_m',1)<.01:cv2.circle(I,tuple(np.int32(x['uv'])),3,(255,0,255),-1,cv2.LINE_AA)
def lab(I,t):
    I=cv2.resize(I,(800,600),interpolation=cv2.INTER_AREA);cv2.rectangle(I,(0,0),(800,34),(20,20,20),-1);cv2.putText(I,t,(8,24),0,.65,(255,255,255),2,cv2.LINE_AA);return I
cap=cv2.VideoCapture(f'{C}/rgb.mp4');out='ep32_cupfix_before_after.mp4';N=len(d['Ns'])
ff=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','2400x600','-r','20','-i','-','-c:v','libx264','-pix_fmt','yuv420p','-crf','20',out],stdin=subprocess.PIPE)
i=0
while True:
    ok,im=cap.read()
    if not ok:break
    p=min(max(i-1,0),N-1);A=im.copy();X=im.copy();Y=im.copy()
    for I,Ts,Tq,nm,col in ((X,d['Ms'],d['Mc'],'merged',(60,170,255)),(Y,d['Ms'],d['Mc2'],'merged',(70,220,70))):
        draw(I,Tq[p],MC,(220,150,60));draw(I,Ts[p],MS,col);spikes(I,Ts[p],Sp[f'wooden_spoon_{nm}'],(0,255,255));spikes(I,Tq[p],Sp[f'blue_cup_{nm}'],(255,255,0));dets(I,p)
    dets(A,p);cv2.putText(A,f'{i/20:.2f}s',(10,110),0,1.4,(255,255,255),3)
    ff.stdin.write(np.hstack([lab(A,'Original (magenta = detected spike balls)'),lab(X,'BEFORE: merged cup'),lab(Y,'AFTER: cup + point tracks')]).tobytes());i+=1
ff.stdin.close();ff.wait();print(i)
