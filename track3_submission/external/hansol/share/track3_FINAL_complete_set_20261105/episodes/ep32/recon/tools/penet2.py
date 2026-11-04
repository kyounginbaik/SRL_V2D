import numpy as np,trimesh,sys
from trimesh.proximity import signed_distance
KD='../bundles/K32_R10/episode_000032/meshes/';cup=trimesh.load(KD+'blue_cup.obj',force='mesh');sp=trimesh.load(KD+'wooden_spoon.obj',force='mesh');assert sp.is_watertight
P,_=trimesh.sample.sample_surface(cup,12000,seed=0);P=np.asarray(P)
d=np.load('cmp_tracks.npz');fin_s=np.load('mergedw_spoon.npy');fin_c=np.load('cup_lam0.4_sg_rot9_pos7.npy')
fr=list(range(60,110))+list(range(125,178))
def pen(Ts,Tc):
    out={}
    for p in fr:
        Rel=np.linalg.inv(Ts[p])@Tc[p];q=P@Rel[:3,:3].T+Rel[:3,3]   # cup surface points in spoon body frame
        near=np.linalg.norm(q-sp.centroid,axis=1)<.2
        if near.sum()==0:out[p]=(0,0);continue
        s=signed_distance(sp,q[near]);out[p]=(1000*max(s.max(),0),int((s>0.002).sum()))
    return out
for nm,Ts,Tc in (('old K32_R10',d['T0'],d['Oc']),('Hansol',d['Ths'],d['Thc']),('FINAL RP2',fin_s,fin_c)):
    o=pen(Ts,Tc);D=np.array([o[p][0] for p in fr]);n=np.array([o[p][1] for p in fr])
    print('== %-12s frames >5mm: %2d  >10mm: %2d  max %.0f mm'%(nm,(D>5).sum(),(D>10).sum(),D.max()))
    for a,b in ((60,110),(125,178)):
        print('   %.2f-%.2fs max mm per 0.25s:'%(a/20,b/20),' '.join(f'{max(o[p][0] for p in range(s,min(s+5,b))):3.0f}' for s in range(a,b,5)))
