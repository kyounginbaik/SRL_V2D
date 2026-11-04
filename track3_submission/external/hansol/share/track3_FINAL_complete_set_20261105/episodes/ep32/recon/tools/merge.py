"""Joint spike-model + pose solve for ep32 (GT-free): masks, stereo, spike pixels, user bowl marks (spoon), temporal prior.
Candidates from old, Hansol, our earlier track, Jisu, plus roll variants. usage: merge.py spoon|cup OUT.npz HOLD(0|1) [workers] [lam]
HOLD=1 keeps back spike detections on p%3==1 and every third mark for testing."""
import sys,json,numpy as np
OBJ,OUT,HOLD=sys.argv[1],sys.argv[2],int(sys.argv[3]);WK=int(sys.argv[4]) if len(sys.argv)>4 else 16;LAM=float(sys.argv[5]) if len(sys.argv)>5 else .1
sys.argv=['r','x','balls=1'];exec(open('refine.py' if OBJ=='spoon' else 'refine_cup.py').read().split("if __name__")[0])
from scipy.optimize import least_squares
import time as _t,os
MW=float(os.environ.get('MW','1'))
def rd(p):
    for i in range(3):
        x=cv2.imread(p,-1)
        if x is not None:return x
        _t.sleep(2+i)
    raise IOError('cannot read '+p)
from scipy.spatial.transform import Slerp
d_=np.load('cmp_tracks.npz');old,han,ours=(d_['T0'],d_['Ths'],d_['Ns']) if OBJ=='spoon' else (d_['Oc'],d_['Thc'],d_['Nc'])
assert np.allclose(old,T0);N=len(T0);KEEP=21 if OBJ=='spoon' else 6
c0=v-v.mean(0);U=np.linalg.eigh(c0.T@c0)[1];LONG=U[:,2]
BOWL=np.array([0,0,-1.]);ANN={}
if OBJ=='spoon':
    ANN={int(k):x for k,x in json.load(open('marks_ep32.json')).items()}
    MH={t for i,t in enumerate(sorted(ANN)) if i%3==0} if HOLD else set()
def hold(p):return HOLD and p%3==1
def rollT(T,a):
    E_=R.from_rotvec(LONG*np.radians(a)).as_matrix();O=np.eye(4);O[:3,:3]=E_;O[:3,3]=com-E_@com;return T@O
def markcost(T,t):
    if t not in ANN or t in MH:return 0.
    nc=T[:3,:3]@BOWL;cen=T[:3,:3]@com+T[:3,3];vw=-cen/np.linalg.norm(cen);f=nc@vw;mk=ANN[t]
    if mk=='out':return MW*max(0,.6-f)*.6
    if mk=='in':return MW*max(0,f+.6)*.6
    p1=cen+.02*nc;a=np.array([p1[0]/p1[2]-cen[0]/cen[2],p1[1]/p1[2]-cen[1]/cen[2]]);a/=np.linalg.norm(a);b=np.array(mk,float);b/=np.linalg.norm(b);return MW*(1-a@b)*.4
def alldet(p):
    x=DET[p+1] if p+1<len(DET) else [];c=T0[p][:3,:3]@com+T0[p][:3,3];return [np.asarray(q['uv'],float) for q in x if np.linalg.norm(np.asarray(q['camera_point'])-c)<.25]
DETS=[alldet(p) for p in range(N-1)]
def uvof(T,b):
    p=b@T[:,:3,:3].transpose(0,2,1)+T[:,:3,3] if b.ndim==1 else None
    return p
def fit_spikes(T,maxk=8,thr=6.,minn=25):
    fr=[p for p in range(N-1) if not hold(p) and DETS[p]];F=np.array([p for p in fr for _ in DETS[p]]);UV=np.array([q for p in fr for q in DETS[p]]);left=np.ones(len(F),bool);S=[];rs=np.random.default_rng(0);info=[]
    Rt=T[F,:3,:3];tt=T[F,:3,3]
    def rep(b):
        pc=np.einsum('nij,j->ni',Rt,b)+tt;return np.c_[fx*pc[:,0]/pc[:,2]+cx,fy*pc[:,1]/pc[:,2]+cy]
    while len(S)<maxk and left.sum()>minn:
        idx=np.flatnonzero(left);seeds=rs.choice(idx,min(200,len(idx)),replace=False);best=(0,None)
        for i in seeds:
            f_=F[i];T1=T[f_];cz=(T1[:3,:3]@com+T1[:3,3])[2]
            for dz in np.linspace(-.12,.12,25):
                z=cz+dz;x=np.array([(UV[i,0]-cx)/fx*z,(UV[i,1]-cy)/fy*z,z]);b=T1[:3,:3].T@(x-T1[:3,3])
                if np.linalg.norm(b-com)>.22:continue
                e=np.linalg.norm(rep(b)-UV,axis=1);n=((e<thr)&left).sum()
                if n>best[0]:best=(n,b)
        if best[1] is None or best[0]<minn:break
        b=best[1]
        for _ in range(4):
            e=np.linalg.norm(rep(b)-UV,axis=1);a=(e<thr)&left
            b=least_squares(lambda q:((rep(q)-UV)[a]).ravel(),b,loss='huber',f_scale=3.,max_nfev=30).x
        e=np.linalg.norm(rep(b)-UV,axis=1);a=(e<thr*1.5)&left;left&=~a;S.append(b);info.append((int(a.sum()),round(float(np.median(e[a])),2)))
    return np.array(S),info
def dcost(T,dat,bon,t):
    sc,iou,dz,bl=parts(T,*dat);c=sc+min(dz,.05)/.02*.15+markcost(T,t)
    if bon and np.isfinite(bl):c+=bl/30*.7
    return c
def getdat(p):
    Mk,h,D,det=load(p+1);c=T0[p][:3,:3]@com+T0[p][:3,3];det=[q for q in det if np.linalg.norm(np.asarray(q['camera_point'])-c)<.25];return Mk,h,D,det
ROLLS=[-150,-120,-90,-60,-30,30,60,90,120,150,180]
def cands(p,cur):
    L=[cur[p],old[p],han[p],ours[p],TJ[p]]
    L+=[rollT(cur[p],a) for a in ROLLS]+[rollT(han[p],a) for a in (-60,-30,30,60)]
    for q in (p-1,p+1):
        if KEEP<=q<N-1:L.append(cur[q])
    return L
def evalc(a):
    p,cur=a;dat=getdat(p);L=cands(p,cur);return p,L,[dcost(T,dat,not hold(p),p+1) for T in L]
def step(a,b):return np.degrees(R.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()),100*np.linalg.norm(a[:3,3]-b[:3,3])
def mid(a,b):
    s=Slerp([0,1],R.from_matrix(np.stack([a[:3,:3],b[:3,:3]])));O=np.eye(4);O[:3,:3]=s(.5).as_matrix();O[:3,3]=(a[:3,3]+b[:3,3])/2;return O
def sweep1(a):
    p,Tp,Tm=a;dat=getdat(p);bon=not hold(p)
    def c(x):
        O=app(Tp,x);r,t=step(Tm,O);return dcost(O,dat,bon,p+1)+LAM*((r/10)**2+.5*t**2)
    r=minimize(c,np.zeros(6),method='Powell',bounds=[(-4,4)]*3+[(-3,3)]*3,options=dict(maxfev=220,xtol=.02,ftol=1e-3));return p,app(Tp,r.x)
def final(a):
    p,X=a;dat=getdat(p);return [parts(old[p],*dat),parts(han[p],*dat),parts(ours[p],*dat),parts(X,*dat)]
if __name__=='__main__':
    T=ours.copy();T[:KEEP]=old[:KEEP];fr=list(range(KEEP,N-1));w=LAM;log=[]
    for rnd in range(3):
        B,info=fit_spikes(T);print('round',rnd,'spikes',len(B),info,np.round(B,3).tolist(),flush=True)
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
    B,info=fit_spikes(T);print('final spikes',len(B),info,np.round(B,3).tolist(),flush=True)
    T[N-1]=T[N-2]@np.linalg.inv(T0[N-2])@T0[N-1]
    P=Pool(WK);Q=np.array(P.map(final,[(p,T[p]) for p in range(N-1)],chunksize=4));np.savez(OUT,T=T,parts=Q,B=B);print('DONE')
