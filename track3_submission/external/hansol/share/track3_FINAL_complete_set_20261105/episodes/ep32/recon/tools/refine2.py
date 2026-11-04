"""Temporal spoon solve for ep32: DP over per-frame candidates, then red-black sweeps with a second-difference prior.
Frames p<35 stay on K32_R10 (leaderboard-selected start roll). usage: refine2.py OUT.npz LAMBDA HOLD(0|1) [workers]. HOLD=1 drops ball terms on p%3==1 for testing."""
import sys,numpy as np
OUT,LAM,HOLD,WK=sys.argv[1],float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]) if len(sys.argv)>4 else 16
sys.argv=['r','x','balls=1'];exec(open('refine.py').read().split("if __name__")[0])
from scipy.spatial.transform import Slerp
b0=np.load('refit_b0.npz')['T'];b1=np.load('refit_b1.npz')['T'];N=len(T0);KEEP=35
def ballon(p):return not(HOLD and p%3==1)
def getdat(p):
    i=p+1;Mk,h,D,det=load(i);c=T0[p][:3,:3]@com+T0[p][:3,3];det=[d for d in det if np.linalg.norm(np.asarray(d['camera_point'])-c)<.15];return Mk,h,D,det
def dcost(T,dat,bon):
    sc,iou,dz,bl=parts(T,*dat);c=sc+min(dz,.05)/.02*.15
    if bon and np.isfinite(bl):c+=bl/30*.5
    return c
def cands(p):
    L=[T0[p],TJ[p],b0[p]]
    if ballon(p):L.append(b1[p])
    for q in (p-1,p+1):
        if KEEP<=q<N-1:L+=[b0[q]]+([b1[q]] if ballon(q) else [])
    return L
def evalc(p):
    dat=getdat(p);L=cands(p);return p,L,[dcost(T,dat,ballon(p)) for T in L]
def step(a,b):return np.degrees(R.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()),100*np.linalg.norm(a[:3,3]-b[:3,3])
def mid(a,b):
    s=Slerp([0,1],R.from_matrix(np.stack([a[:3,:3],b[:3,:3]])));O=np.eye(4);O[:3,:3]=s(.5).as_matrix();O[:3,3]=(a[:3,3]+b[:3,3])/2;return O
def sweep1(a):
    p,Tp,Tm=a;dat=getdat(p);bon=ballon(p)
    def c(x):
        O=app(Tp,x);r,t=step(Tm,O);return dcost(O,dat,bon)+LAM*((r/10)**2+.5*t**2)
    r=minimize(c,np.zeros(6),method='Powell',bounds=[(-4,4)]*3+[(-3,3)]*3,options=dict(maxfev=220,xtol=.02,ftol=1e-3));return p,app(Tp,r.x)
def final(a):
    p,X=a;dat=getdat(p);return [parts(T0[p],*dat),parts(X,*dat)]
if __name__=='__main__':
    P=Pool(WK);fr=list(range(KEEP,N-1));res=P.map(evalc,fr,chunksize=4)
    C={p:(L,np.array(c)) for p,L,c in res};w=LAM
    best=C[KEEP][1]+[w*((step(T0[KEEP-1],T)[0]/10)**2+.5*step(T0[KEEP-1],T)[1]**2) for T in C[KEEP][0]];bp={}
    for p in fr[1:]:
        L,c=C[p];Lp=C[p-1][0];nb=np.zeros(len(L));bk=np.zeros(len(L),int)
        for j,T in enumerate(L):
            q=[best[k]+w*((step(U,T)[0]/10)**2+.5*step(U,T)[1]**2) for k,U in enumerate(Lp)];bk[j]=int(np.argmin(q));nb[j]=q[bk[j]]+c[j]
        bp[p]=bk;best=nb
    j=int(np.argmin(best));T=T0.copy()
    for p in fr[::-1]:
        T[p]=C[p][0][j]
        if p>KEEP:j=bp[p][j]
    for s in range(4):
        for par in (0,1):
            jobs=[(p,T[p],mid(T[p-1],T[p+1])) for p in fr if p%2==par]
            for p,O in P.map(sweep1,jobs,chunksize=4):T[p]=O
        print('sweep',s,flush=True)
    T[N-1]=T[N-2]@np.linalg.inv(T0[N-2])@T0[N-1]
    Q=np.array(P.map(final,[(p,T[p]) for p in range(N-1)],chunksize=4));np.savez(OUT,T=T,T0=T0,parts=Q);print('DONE')
