import sys,numpy as np,cv2
sys.argv=['r','x','balls=1'];exec(open('refine.py').read().split("if __name__")[0])
SC=1.0;d=np.load('cmp_tracks.npz');mc=trimesh.load(f'{KD}/meshes/blue_cup.obj',force='mesh')
DC=json.load(open(f'{C}/ball_detections_blue_cup.json'))['frames']
print('spoon det per frame',np.bincount([len(x) for x in DET]).tolist(),'cup',np.bincount([len(x) for x in DC]).tolist(),'keys',list(DC[100][0].keys()) if DC[100] else '')
def mk(mm):return np.asarray(mm.vertices),np.asarray(mm.faces),np.asarray(mm.face_normals)
MS=mk(m);MC=mk(mc)
def draw(I,T,M,col):
    v_,f_,n_=M;p=v_@T[:3,:3].T+T[:3,3];u=proj(p).round().astype(np.int32);fz=p[f_,2].mean(1);nc=n_@T[:3,:3].T;cc=p[f_].mean(1);vw=-cc/np.linalg.norm(cc,axis=1,keepdims=True)
    for t in np.argsort(-fz):
        s=(nc[t]*vw[t]).sum()
        if s>0:cv2.fillConvexPoly(I,u[f_[t]],tuple(int(c*(.25+.75*s**2)) for c in col))
cap=cv2.VideoCapture(f'{C}/rgb.mp4');Fr=[]
while True:
    ok,im=cap.read()
    if not ok:break
    Fr.append(im)
def crop(I,c,r=110):
    x=int(np.clip(c[0]-r,0,W-2*r));y=int(np.clip(c[1]-r,0,H-2*r));return cv2.resize(I[y:y+2*r,x:x+2*r],(330,330),interpolation=cv2.INTER_CUBIC)
rows=[]
for p in (5,30,105,125,200,225):
    Tc,Ts=d['Nc'][p],d['Ns'][p];cc=proj((Tc[:3,:3]@np.asarray(mc.vertices).mean(0)+Tc[:3,3])[None])[0];cs=proj((Ts[:3,:3]@com+Ts[:3,3])[None])[0]
    Rc=np.zeros((H,W,3),np.uint8);draw(Rc,Tc,MC,(220,190,120));Rs=np.zeros((H,W,3),np.uint8);draw(Rs,Ts,MS,(200,230,250))
    A=crop(Fr[p+1],cc);B=crop(Rc,cc);D=crop(Fr[p+1],cs,150);E=crop(Rs,cs,150);cv2.putText(A,f'p{p}',(6,26),0,.8,(255,255,255),2)
    rows.append(np.hstack([A,B,D,E]))
cv2.imwrite('sp1.jpg',np.vstack(rows))
# mesh bumps: distance from a heavily smoothed copy
for nm,mm in (('spoon',m),('cup',mc)):
    print(nm,'verts',len(mm.vertices),'bounds cm',np.round(100*mm.extents,1).tolist(),'watertight',mm.is_watertight)
