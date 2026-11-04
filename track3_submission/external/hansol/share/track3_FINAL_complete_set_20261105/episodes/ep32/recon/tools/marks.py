import json,re,numpy as np,trimesh
from pathlib import Path
O=Path('/data1/hansol/retarget32/out/marks');ANN={int(k):v for k,v in json.load(open(O/'ann_ep32_flat.json')).items()}
d=np.load('cmp_tracks.npz');Ths,Ns,T0=d['Ths'],d['Ns'],d['T0']
V14=np.load('/data1/SRL_V2D/challenge_reconstruction/reconstruction_recovery_20261026/selected20_final_v14_g17/episode_000032/object_poses.npz');Tv=V14['camera_from_object'][:,list(V14['names']).index('wooden_spoon')]
m=trimesh.load('../bundles/K32_R10/episode_000032/meshes/wooden_spoon.obj',force='mesh');v=np.asarray(m.vertices);c=v-v.mean(0);U=np.linalg.eigh(c.T@c)[1];print('pca',np.round(U,3).tolist())
def ref(n):
    r={}
    for l in (O/f'{n}.txt').read_text().splitlines():
        g=re.match(r"^(\d+) .*'facing': ([-\d.]+), 'arrow_err': ([\d.]+|None), 'ok': (True|False)",l.strip())
        if g:r[int(g[1])]=(float(g[2]),None if g[3]=='None' else float(g[3]),g[4]=='True')
    return r
RH=ref('ours_R32_HGA');RV=ref('v14_g17')
def calc(T,n,lag,t):
    X=T[max(t-lag,0)];nc=X[:3,:3]@n;cen=X[:3,:3]@v.mean(0)+X[:3,3];vw=-cen/np.linalg.norm(cen);f=nc@vw
    p0=cen;p1=cen+.02*nc;a=np.array([p1[0]/p1[2]-p0[0]/p0[2],p1[1]/p1[2]-p0[1]/p0[2]]);return f,a/np.linalg.norm(a)
best=None
cands=[s*U[:,i] for i in range(3) for s in (1,-1)]
for n in cands:
    for lag in (0,1):
        e=np.mean([abs(calc(Ths,n,lag,t)[0]-RH[t][0]) for t in RH]);ae=[]
        for t,mk in ANN.items():
            if isinstance(mk,list) and RH[t][1] is not None:
                f,a=calc(Ths,n,lag,t);b=np.array(mk)/np.linalg.norm(mk);ae.append(abs(np.degrees(np.arccos(np.clip(a@b,-1,1)))-RH[t][1]))
        s=e*100+np.mean(ae)
        if best is None or s<best[0]:best=(s,n,lag,e,np.mean(ae))
print('best match to their numbers: facing err %.3f, arrow err %.1f deg, lag %d, n %s'%(best[3],best[4],best[2],np.round(best[1],3)));n,lag=best[1],best[2]
for th in (30,45,60):
  pass
def score(T,thf=.6,tha=45):
    ok={}
    for t,mk in ANN.items():
        f,a=calc(T,n,lag,t)
        if mk=='out':ok[t]=f>thf
        elif mk=='in':ok[t]=f<-thf
        else:
            b=np.array(mk)/np.linalg.norm(mk);ok[t]=np.degrees(np.arccos(np.clip(a@b,-1,1)))<tha
    return ok
W=[(40,70,'2.0-3.5'),(73,94,'3.65-4.7'),(97,130,'4.85-6.5'),(133,154,'6.65-7.7'),(157,209,'7.85-10.45')]
for nm,T,rr in (('hansol',Ths,RH),('v14',Tv,RV),('old R10',T0,None),('ours new',Ns,None)):
    o=score(T);row=' '.join(f"{w[2]}:{sum(o[t] for t in o if w[0]<=t<=w[1])}/{sum(1 for t in o if w[0]<=t<=w[1])}" for w in W)
    ag=f"; agrees with their ok-flags {sum(o[t]==rr[t][2] for t in o)}/58, their total {sum(x[2] for x in rr.values())}" if rr else ''
    print(nm,'total',sum(o.values()),'/58 |',row,ag)
    print('   misses',[t for t in sorted(o) if not o[t]])
