import numpy as np, trimesh, kitx
from scipy.spatial.transform import Rotation as Rot
p='/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/canonical/public_heldout/episode_000000/objects/white_pot_lid/mesh.obj'
m=trimesh.load(p,force='mesh'); v=m.vertices; print('verts',len(v),'faces',len(m.faces),'watertight',m.is_watertight)
c=v.mean(0); w,U=np.linalg.eigh(np.cov((v-c).T)); print('eig',w, 'axes',U.T)
ax=U[:,0]; h=(v-c)@ax; print('height range',h.min(),h.max(), 'skew', ((h-h.mean())**3).mean()/h.std()**3)
# polar
e1=U[:,2]; e2=np.cross(ax,e1); x=(v-c)@e1; y=(v-c)@e2; r=np.hypot(x,y); th=np.degrees(np.arctan2(y,x))
hi=h>np.percentile(h,90) if ((h-h.mean())**3).mean()>0 else h<np.percentile(h,10)
print('handle side pts',hi.sum(),'r range',r[hi].min(),r[hi].max())
# handle direction via PCA of top points in-plane
P=np.c_[x[hi],y[hi]]; ww,uu=np.linalg.eigh(np.cov(P.T)); print('handle in-plane dir deg',np.degrees(np.arctan2(uu[1,1],uu[0,1])), 'elong',ww)
# radial profile asymmetry: rim radius vs angle
for a in range(-180,180,30):
    s=(th>=a)&(th<a+30)&(np.abs(h-np.median(h))<0.01); print(a, 'rim r max %.3f'%(r[s].max() if s.any() else np.nan), s.sum())
# scorer side: registration rotation; where does our handle go relative to scan handle
v2,f2,anc,axr,od=kitx.ref_geom('white_pot_lid',2); cv,cf=kitx.sub_mesh(p)
ca,rot=kitx.M._register_object_body((cv,cf),(v2,f2),anc,1)
# map our handle dir into scan frame
cc,_=kitx.M._surface_moments(cv,cf)
hd=uu[0,1]*e1+uu[1,1]*e2; hd_s=rot@hd; ax_s=rot@ax
vr=v2-v2.mean(0); hr=vr@axr; sk=((hr-hr.mean())**3).mean()
hir=hr>np.percentile(hr,90) if sk>0 else hr<np.percentile(hr,10)
f1=np.cross(axr,[1,0,0]); f1/=np.linalg.norm(f1); f2_=np.cross(axr,f1)
Pr=np.c_[vr[hir]@f1,vr[hir]@f2_]; wr,ur=np.linalg.eigh(np.cov(Pr.T)); 
print('scan handle dir in (f1,f2) deg',np.degrees(np.arctan2(ur[1,1],ur[0,1])),'ours mapped',np.degrees(np.arctan2(hd_s@f2_,hd_s@f1)),'axis agreement deg',np.degrees(np.arccos(abs(ax_s@axr))))
