"""Silhouette radial-scale test of a cup mesh: best radial scale s (about the cup axis) vs the SAM2 mask, frames where the
object is unoccluded by the other object. No GT. usage: radial_fit.py ROOT POSES_DIR NAME FRAMES(a:b:c)"""
import sys, json, numpy as np, cv2
from pathlib import Path
from nestlib import *
root,pdir,name=sys.argv[1],sys.argv[2],sys.argv[3]; a,b,c=[int(x) for x in sys.argv[4].split(':')]
P=np.load(Path(pdir)/'object_poses.npz'); names=[str(n) for n in P['names']]; C=P['camera_from_object']
meshes=json.loads((Path(pdir)/'meshes.json').read_text()) if (Path(pdir)/'meshes.json').exists() else {}
m0=load_mesh(meshes.get(name,f'{root}/objects/{name}/mesh.obj')); ax,ce,_=container_axis(m0)
v=np.asarray(m0.vertices); h=(v-ce)@ax; rad=(v-ce)-np.outer(h,ax)
S=np.round(np.arange(0.92,1.081,0.01),3)
rs=[]
for s in S:
    m=m0.copy(); m.vertices=ce+np.outer(h,ax)+s*rad; rs.append(Renderer([m],*intrinsics(root)))
K,W,H=intrinsics(root); i=names.index(name); res=[]
for f in range(a,b,c):
    M=read_mask(root,name,f)
    if M is None or M.sum()<500: continue
    ex=np.zeros_like(M)
    for n in names:
        for l in ('23' if n==name else '123'):
            x=read_mask(root,n,f,l)
            if x is not None and not (n!=name and l=='1' and False): ex|=x
    ex=cv2.dilate(ex.astype(np.uint8),np.ones((7,7),np.uint8))>0; ex&=~M
    ys,xs=np.nonzero(M); roi=(max(0,xs.min()-40),max(0,ys.min()-40),min(W,xs.max()+40),min(H,ys.max()+40)); sl=(slice(roi[1],roi[3]),slice(roi[0],roi[2]))
    # skip frames where the other object's mask touches this one (occlusion)
    near=cv2.dilate(M.astype(np.uint8),np.ones((9,9),np.uint8))>0
    occ=any((read_mask(root,n,f) & near).sum()>0 for n in names if n!=name) or ((ex&near).sum()>50)
    if occ: continue
    ious=[]
    for r in rs:
        lab,_,_=r.render([C[f,i]],roi); R=lab==0; Mm=M[sl]; E=ex[sl]
        ious.append(((R&Mm)&~E).sum()/max(1,((R|Mm)&~E).sum()))
    ious=np.array(ious); res.append((f,float(S[ious.argmax()]),float(ious.max()),float(ious[list(S).index(1.0)]), int(M.sum())))
if res:
    s=np.array([r[1] for r in res]); print(f'{root.split("/")[-2]}/{root.split("/")[-1]} {name} frames {len(res)} best radial scale median {np.median(s):.3f} (p25 {np.percentile(s,25):.3f} p75 {np.percentile(s,75):.3f}) iou@best {np.median([r[2] for r in res]):.3f} iou@1 {np.median([r[3] for r in res]):.3f}')
else: print(root,name,'no unoccluded frames')
