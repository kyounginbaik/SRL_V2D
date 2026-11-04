"""Cup re-solve with cotracker point tracks added (GT-free). usage: merge_cup2.py OUT.npz HOLD(0|1) [workers] [lam] [ctw]
HOLD=1: fit on even-index tracks + spikes on p%3!=1; report on odd tracks / held frames."""
import sys,os,numpy as np
OUT,HOLD=sys.argv[1],int(sys.argv[2]);WK=int(sys.argv[3]) if len(sys.argv)>3 else 14;LAM=float(sys.argv[4]) if len(sys.argv)>4 else .1;CTW=float(sys.argv[5]) if len(sys.argv)>5 else .5
sys.argv=['merge.py','cup',OUT,str(HOLD),str(WK),str(LAM)]
exec(open('merge.py').read().rsplit("\nif __name__",1)[0])
ct=np.load(f'{C}/point_tracks_cotracker.npz',allow_pickle=True);cupk=np.flatnonzero(ct['object_ids']==0);TR=ct['tracks'][:,cupk];VIS=ct['visible'][:,cupk];NK=len(cupk)
FIT=np.array([k%2==0 for k in range(NK)]) if HOLD else np.ones(NK,bool);EVAL=~FIT if HOLD else FIT
RI=trimesh.ray.ray_pyembree.RayMeshIntersector(m) if trimesh.ray.has_embree else trimesh.ray.ray_triangle.RayMeshIntersector(m)
BP=None;BPok=None
def lift(T,goodmask):
    body=[[] for _ in range(NK)]
    for p in np.flatnonzero(goodmask):
        i=p+1
        if i>=N:continue
        iT=np.linalg.inv(T[p]);vv=VIS[i];uv=TR[i];dirs=np.c_[(uv[:,0]-cx)/fx,(uv[:,1]-cy)/fy,np.ones(NK)];dirs/=np.linalg.norm(dirs,axis=1,keepdims=True)
        loc,ray,_=RI.intersects_location(np.tile(iT[:3,3],(vv.sum(),1)),(dirs@iT[:3,:3].T)[vv],multiple_hits=False)
        for k,j in zip(ray,loc):body[np.flatnonzero(vv)[k]].append(j)
    B_=np.zeros((NK,3));ok=np.zeros(NK,bool)
    for k in range(NK):
        if len(body[k])>=8:
            bb=np.array(body[k]);mm=np.median(bb,axis=0);sp=np.median(np.linalg.norm(bb-mm,axis=1))
            if sp<.02:B_[k]=mm;ok[k]=True
    return B_,ok
def ctres(T,p,sel):
    i=p+1
    if i>=N or BP is None:return np.array([])
    vv=VIS[i]&BPok&sel
    if vv.sum()<4:return np.array([])
    u=proj(BP[vv]@T[:3,:3].T+T[:3,3]);return np.linalg.norm(u-TR[i][vv],axis=1)
_dc=dcost
def dcost(T,dat,bon,t):
    e=ctres(T,t-1,FIT);c=_dc(T,dat,bon,t)
    if len(e):c+=CTW*np.mean(np.minimum(e,25.))/25.
    return c
def final(a):
    p,X=a;dat=getdat(p);e=ctres(X,p,EVAL);e0=ctres(old[p],p,EVAL);eh=ctres(han[p],p,EVAL);em=ctres(M0[p],p,EVAL)
    f=lambda z:float(np.median(z)) if len(z) else np.nan
    return [list(parts(old[p],*dat))+[f(e0)],list(parts(han[p],*dat))+[f(eh)],list(parts(M0[p],*dat))+[f(em)],list(parts(X,*dat))+[f(e)]]
if __name__=='__main__':
    M0=np.load('merged_cup.npy');T=M0.copy();fr=list(range(KEEP,N-1));w=LAM;Q0=np.load('mg_cup_0.npz')['parts']
    good=np.zeros(N,bool);good[:N-1]=Q0[:,3,1]>=.9
    for rnd in range(3):
        BP,BPok=lift(T,good);print('round',rnd,'lifted tracks',int(BPok.sum()),flush=True)
        B,info=fit_spikes(T);print('  spikes',len(B),info,flush=True)
        P=Pool(WK);res=P.map(evalc,[(p,T) for p in fr],chunksize=4);CC={p:(L,np.array(c)) for p,L,c in res}
        best=CC[KEEP][1]+np.array([w*((step(T[KEEP-1],X)[0]/10)**2+.5*step(T[KEEP-1],X)[1]**2) for X in CC[KEEP][0]]);bp={}
        for p in fr[1:]:
            L,c=CC[p];Lp=CC[p-1][0];nb=np.zeros(len(L));bk=np.zeros(len(L),int)
            for j,X in enumerate(L):
                q=best+[w*((step(U_,X)[0]/10)**2+.5*step(U_,X)[1]**2) for U_ in Lp];bk[j]=int(np.argmin(q));nb[j]=q[bk[j]]+c[j]
            bp[p]=bk;best=nb
        j=int(np.argmin(best));T2=T.copy()
        for p in fr[::-1]:
            T2[p]=CC[p][0][j]
            if p>KEEP:j=bp[p][j]
        T=T2
        for s in range(3):
            for par in (0,1):
                jobs=[(p,T[p],mid(T[p-1],T[p+1])) for p in fr if p%2==par]
                for p,O in P.map(sweep1,jobs,chunksize=4):T[p]=O
        P.close();P.join();print('round done',rnd,flush=True)
    BP,BPok=lift(T,good);B,info=fit_spikes(T);T[N-1]=T[N-2]@np.linalg.inv(T0[N-2])@T0[N-1];np.save(OUT+'.T.npy',T)
    P=Pool(WK);Q=np.array(P.map(final,[(p,T[p]) for p in range(N-1)],chunksize=4));np.savez(OUT,T=T,parts=Q,B=B,BP=BP,BPok=BPok);print('DONE')
