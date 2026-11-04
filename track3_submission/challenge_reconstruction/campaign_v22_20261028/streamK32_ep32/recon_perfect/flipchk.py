import sys,numpy as np,cv2
sys.argv=['r','x','balls=1'];exec(open('refine.py').read().split("if __name__")[0])
Tn=np.load('full_0.1.npz')['T'];rc=R.from_matrix(T0[:,:3,:3]);rn=R.from_matrix(Tn[:,:3,:3]);ch=np.degrees((rc.inv()*rn).magnitude());chJ=np.degrees((R.from_matrix(TJ[:,:3,:3]).inv()*rn).magnitude())
print('per 10 frames, max change vs R10 / vs Jisu')
print(' '.join(f'{i}:{ch[i:i+10].max():.0f}/{chJ[i:i+10].max():.0f}' for i in range(30,259,10)))
top=[int(i) for i in np.argsort(-ch)[:0]]
spans=[p for p in range(35,258) if ch[p]>60];print('frames >60 deg',spans)
cap=cv2.VideoCapture(f'{C}/rgb.mp4');Fr=[]
while True:
    ok,im=cap.read()
    if not ok:break
    Fr.append(im)
SC=1.0;nf=np.asarray(m.face_normals)
def shade(T):
    p=v@T[:3,:3].T+T[:3,3];u=proj(p).round().astype(np.int32);I=np.zeros((H,W,3),np.uint8);fz=p[f,2].mean(1);nc=nf@T[:3,:3].T;cc=p[f].mean(1);vw=-cc/np.linalg.norm(cc,axis=1,keepdims=True)
    for t in np.argsort(-fz):
        s=(nc[t]*vw[t]).sum()
        if s>0:cv2.fillConvexPoly(I,u[f[t]],(int(60+190*s),int(120+130*s),int(170+85*s)))
    return I
def crop(im,T):
    c=proj((T[:3,:3]@com+T[:3,3])[None])[0];x=int(np.clip(c[0]-200,0,W-400));y=int(np.clip(c[1]-150,0,H-300));return im[y:y+300,x:x+400]
pick=[140,175,178,182,184,189]
rows=[]
for p in pick:
    a=Fr[p+1].copy();cv2.putText(a,f'p{p} ch{ch[p]:.0f}',(10,40),0,1.2,(255,255,255),3)
    rows.append(np.hstack([cv2.resize(Fr[p+1][max(0,int(proj((Tn[p][:3,:3]@com+Tn[p][:3,3])[None])[0][1])-150):,:][:300],(0,0),fx=1,fy=1)[:300,:400] if 0 else crop(a,Tn[p]),crop(shade(T0[p]),Tn[p]),crop(shade(Tn[p]),Tn[p])]))
cv2.imwrite('flip.jpg',np.vstack(rows));print(pick)
