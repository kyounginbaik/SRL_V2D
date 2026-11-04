import sys, numpy as np, cv2
sys.argv=[sys.argv[0],'coax']+sys.argv[1:]
exec(open('scan_ep10.py').read().split('H=np.round')[0])
f=int(sys.argv[2]); eps=float(sys.argv[3]); Tb,Te=C[f,ib],C[f,ie]
mB=read_mask(R,'blue_cup',f); mE=read_mask(R,'beige_cup',f)
ys,xs=np.nonzero(mB|mE); roi=(xs.min()-30,ys.min()-30,xs.max()+30,ys.max()+30); sl=(slice(roi[1],roi[3]),slice(roi[0],roi[2]))
cap=cv2.VideoCapture(R+'/rgb.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,img=cap.read(); img=img[sl]
def col(lb):
    o=img.copy()//2; o[lb==0]=(200,80,0); o[lb==1]=(120,220,240); return o
gt=np.where(mE[sl],1,np.where(mB[sl],0,-1)); ims=[img,col(gt)]
lab,_=render_nested(rB,rE,Tb,Te,roi,ab,cb,hb.max(),eps); ims.append(col(lab))
for h in [0.025,0.045,0.065]:
    lab,_=render_nested(rB,rE,Tb,beige_pose(Tb,Te,h),roi,ab,cb,hb.max(),eps); ims.append(col(lab))
cv2.imwrite(f'../out/panels_{f}_nested{int(eps*1000)}.png',np.hstack(ims))
