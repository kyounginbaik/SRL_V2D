import sys,numpy as np,cv2
sys.argv=['r','x','balls=1'];exec(open('refine.py').read().split("if __name__")[0])
Tn=np.load('full_0.1.npz')['T'];cap=cv2.VideoCapture(f'{C}/rgb.mp4');Fr=[];
while True:
    ok,im=cap.read()
    if not ok:break
    Fr.append(im)
SC=1.0
def cont(im,T,col):
    c,_=cv2.findContours(sil(T).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);cv2.drawContours(im,c,-1,col,2)
def sheet(ps,nm):
    o=[]
    for p in ps:
        im=Fr[p+1].copy();cont(im,T0[p],(0,0,255));cont(im,Tn[p],(0,255,0))
        for b in proj(B@Tn[p][:3,:3].T+Tn[p][:3,3]):cv2.circle(im,tuple(np.int32(b)),6,(0,255,0),1)
        for d in (DET[p+1] if p+1<len(DET) else []):cv2.circle(im,tuple(np.int32(d['uv'])),4,(255,0,255),-1)
        cv2.putText(im,f'pose {p} {(p+1)/20:.2f}s',(10,40),0,1.2,(255,255,255),3);o.append(cv2.resize(im,(640,int(640*H/W))))
    cv2.imwrite(nm,np.vstack([np.hstack(o[i:i+4]) for i in range(0,len(o),4)]))
sheet([87,89,90,92,150,151,153,155,69,72,98,134,177,179,181,200],'final_check.jpg')
