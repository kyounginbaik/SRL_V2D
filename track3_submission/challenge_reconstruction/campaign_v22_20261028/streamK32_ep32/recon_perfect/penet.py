import numpy as np,trimesh
KD='../bundles/K32_R10/episode_000032/meshes/'
cup=trimesh.load(KD+'blue_cup.obj',force='mesh');sp=trimesh.load(KD+'wooden_spoon.obj',force='mesh');V=np.asarray(sp.vertices)
cv=np.asarray(cup.vertices);c0=cv-cv.mean(0);U=np.linalg.eigh(c0.T@c0)[1];ax=U[:,2];ctr=cv.mean(0)
z=c0@ax;rad=np.linalg.norm(c0-np.outer(z,ax),axis=1)
if np.median(rad[z<np.percentile(z,10)])>np.median(rad[z>np.percentile(z,90)]):ax=-ax;z=-z   # base end = small radius
zb=np.arange(z.min(),z.max()+.005,.005);rin=np.full(len(zb),np.nan);rout=np.full(len(zb),np.nan)
for i,a in enumerate(zb):
    s=(z>=a)&(z<a+.005)
    if s.sum()>5:rin[i]=np.percentile(rad[s],5);rout[i]=np.percentile(rad[s],95)
ok=np.isfinite(rin);zb,rin,rout=zb[ok],rin[ok],rout[ok];zbase=z.min();ztop=z.max()
print('cup: height %.1f cm, radius bottom %.1f top %.1f cm, wall thickness median %.1f mm'%(100*(ztop-zbase),100*rout[0],100*rout[-1],1000*np.median(rout[3:-1]-rin[3:-1])))
base_in=zbase+0.004  # inner bottom ~4 mm above outer base (estimate)
def pen(Ts,Tc,frames):
    out=[]
    for p in frames:
        Rel=np.linalg.inv(Tc[p])@Ts[p];q=V@Rel[:3,:3].T+Rel[:3,3];qz=(q-ctr)@ax;qr=np.linalg.norm((q-ctr)-np.outer(qz,ax),axis=1)
        k=np.clip(np.searchsorted(zb,qz)-1,0,len(zb)-1);ri=rin[k];ro=rout[k]
        inside=(qz>zbase)&(qz<ztop-.01)  # below rim band
        wall=inside&(qr>ri-.001)&(qr<ro+.001);wd=np.where(wall,np.minimum(qr-ri,ro-qr)+.001,0)
        base=(qz<base_in)&(qz>zbase-.01)&(qr<ro[0]);bd=np.where(base,base_in-qz,0)
        cav=inside&(qr<ri-.001);outside=inside&(qr>ro+.001)
        out.append((1000*wd.max(),1000*bd.max(),int(cav.sum()),int(outside.sum()),int(wall.sum())))
    return np.array(out)
d=np.load('cmp_tracks.npz');fin_s=np.load('mergedw_spoon.npy');fin_c=np.load('cup_lam0.4_sg_rot9_pos7.npy')
fr=list(range(0,258))
for nm,Ts,Tc in (('old K32_R10',d['T0'],d['Oc']),('Hansol',d['Ths'],d['Thc']),('FINAL RP2',fin_s,fin_c)):
    P=pen(Ts,Tc,fr);print('==',nm,'| frames with wall pen >3mm: %d, >8mm: %d; max wall %.0f mm; base pen frames %d max %.0f mm'%((P[:,0]>3).sum(),(P[:,0]>8).sum(),P[:,0].max(),(P[:,1]>2).sum(),P[:,1].max()))
    for a,b in ((60,110),(125,175)):
        print('   %.2f-%.2fs per 0.25s max wall mm:'%(a/20,b/20),' '.join(f'{P[s:s+5,0].max():3.0f}' for s in range(a,b,5)))
        print('   %.2f-%.2fs per 0.25s cavity/outside/wall verts:'%(a/20,b/20),' '.join(f'{P[s:s+5,2].max()}/{P[s:s+5,3].max()}/{P[s:s+5,4].max()}' for s in range(a,b,5)))
