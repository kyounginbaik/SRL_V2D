import sys, numpy as np, cv2
from nestlib import *
R='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/evaluation/episode_000010'
f=int(sys.argv[1]); P=np.load(sys.argv[2] if len(sys.argv)>2 else R+'/exports/final_v5/object_poses.npz'); names=[str(n) for n in P['names']]; C=P['camera_from_object']
mb=load_mesh(R+'/objects/blue_cup/mesh.obj'); me=load_mesh(R+'/objects/beige_cup/mesh.obj')
K,Wd,Ht=intrinsics(R); rend=Renderer([mb,me],K,Wd,Ht)
lab,z,_=rend.render([C[f,names.index('blue_cup')],C[f,names.index('beige_cup')]],bias=[0,0.006])
cap=cv2.VideoCapture(R+'/rgb.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,img=cap.read()
print(img.shape if ok else None)
o=img.copy()
for i,col in ((0,(255,0,0)),(1,(0,200,255))):
    m=(lab==i).astype(np.uint8); cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(o,cs,-1,col,1)
for n,col in (('blue_cup',(255,255,0)),('beige_cup',(0,0,255))):
    m=read_mask(R,n,f).astype(np.uint8); cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(o,cs,-1,col,1)
cv2.imwrite(sys.argv[3] if len(sys.argv)>3 else f'../out/ov_{f}.png',o[300:760,300:760])
