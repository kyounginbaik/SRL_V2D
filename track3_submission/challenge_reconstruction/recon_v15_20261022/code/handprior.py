"""Hand-rigid-hold prior for the sweeper (GT-free). Loads stage A (world poses, hands) and stage B metric poses."""
import sys, os, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); W=os.path.dirname(os.path.dirname(HERE)); sys.path[:0]=[W+'/code/postprocess', W+'/code']
import score_public_tracking as S, evaluate2 as E2, pp
from scipy.spatial.transform import Rotation as R
def load(ep,obj='mini_sweeper'):
    d=f'{W}/outputs/episode_{ep:06d}/{obj}/result_slam_gravity_aligned/result.npz'; z=np.load(d)
    A=S.stageA_poses(ep,obj); B=E2.stageB_tag(ep,obj,'_metric')
    hands={s:(z[f'hand_{s}_wrist_to_world_transform'].astype(float), z[f'hand_{s}_is_valid'].astype(bool)) for s in ('left','right')}
    return A,B,hands
def grasp(Tc, hands, dmax=0.13, minlen=10):
    """per frame: which hand holds the object (distance wrist->object centre), smoothed; returns (hand index array: -1 none, 0 left, 1 right)"""
    n=len(Tc); c=Tc[:,:3,3]; out=-np.ones(n,int); D=[]
    for s in ('left','right'):
        H,v=hands[s]; d=np.linalg.norm(H[:n,:3,3]-c,axis=1); d[~v[:n]]=9; D.append(d)
    D=np.array(D); best=D.argmin(0); ok=D.min(0)<dmax; out[ok]=best[ok]
    # remove short segments
    i=0
    while i<n:
        j=i
        while j<n and out[j]==out[i]: j+=1
        if out[i]>=0 and j-i<minlen: out[i:j]=-1
        i=j
    return out, D
def predict_rot(Tc, hands, who):
    """rotation prediction: inside each grasp segment, R_obj = R_hand(t) R_hand(s)^-1 R_obj(s-1)."""
    n=len(Tc); Rp=Tc[:,:3,:3].copy(); have=np.zeros(n,bool); i=0
    while i<n:
        if who[i]<0: i+=1; continue
        j=i
        while j<n and who[j]==who[i]: j+=1
        H,_=hands[('left','right')[who[i]]]; s=max(i-1,0); R0=Tc[s,:3,:3]; Hs=H[i,:3,:3]
        for k in range(i,j): Rp[k]=H[k,:3,:3]@Hs.T@R0; have[k]=True
        i=j
    return Rp, have


def ang(Ra, Rb): return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T@Rb)-1)/2,-1,1))))
def twin_select(T, twins, hands, lam_cont=1.0, lam_prior=1.0, dmax=0.13, init=0, prior_seed_lag=1):
    """Viterbi over half-turn twins of one track T (states = twin index).
    cost(t,i) = lam_cont * rot step from (t-1,j) + lam_prior * angle to the hand-propagated rotation inside grasp segments.
    The hand prediction for segment [s,e) uses the selected state's rotation at s-prior_seed_lag, evaluated greedily (forward pass keeps,
    per state, its own predicted rotation)."""
    n=len(T); m=len(twins); X=np.array([[T[t]@S for S in twins] for t in range(n)])  # n x m x 4 x 4
    who,_=grasp(X[:,0],hands,dmax=dmax)
    cost=np.full(m,1e3); cost[init]=0.; back=np.zeros((n,m),int); seed=[None]*m
    for t in range(1,n):
        step=np.array([[ang(X[t,i,:3,:3],X[t-1,j,:3,:3]) for j in range(m)] for i in range(m)])
        tot=cost[None,:]+lam_cont*step; back[t]=tot.argmin(1); new=tot.min(1)
        newseed=[seed[back[t][i]] for i in range(m)]
        if who[t]>=0:
            H=hands[('left','right')[who[t]]][0]
            for i in range(m):
                if who[t-1]!=who[t] or newseed[i] is None:   # segment start: seed from the previous frame of this path
                    newseed[i]=(H[t,:3,:3].copy(), X[t-1,back[t][i],:3,:3].copy())
                Hs,R0=newseed[i]; pred=H[t,:3,:3]@Hs.T@R0; new[i]+=lam_prior*ang(X[t,i,:3,:3],pred)
        else:
            newseed=[None]*m
        cost,seed=new,newseed
    path=np.zeros(n,int); path[-1]=int(cost.argmin())
    for t in range(n-1,0,-1): path[t-1]=back[t][path[t]]
    return X[np.arange(n),path], path, who

def segments(who):
    out=[]; i=0; n=len(who)
    while i<n:
        j=i
        while j<n and who[j]==who[i]: j+=1
        out.append((int(who[i]),i,j)); i=j
    return out
def mean_rot(Rs):
    M=np.sum(Rs,0); U,_,Vt=np.linalg.svd(M); D=np.diag([1,1,np.sign(np.linalg.det(U@Vt))]); return U@D@Vt
def smooth_T(T, half=3):
    from scipy.ndimage import uniform_filter1d
    out=T.copy(); out[:,:3,3]=uniform_filter1d(T[:,:3,3],2*half+1,axis=0,mode='nearest')
    q=R.from_matrix(T[:,:3,:3]).as_quat(); 
    for k in range(1,len(q)):
        if np.dot(q[k],q[k-1])<0: q[k]=-q[k]
    q=uniform_filter1d(q,2*half+1,axis=0,mode='nearest'); out[:,:3,:3]=R.from_quat(q/np.linalg.norm(q,axis=1,keepdims=True)).as_matrix(); return out
def rigid_hold(T, twins, hands, dmax=0.13, hand_half=3, inlier_deg=35., inlier_cm=4., hybrid=False):
    """Inside each grasp segment: object = H(t) @ X with X the robust hand->object transform from the frames of T (twin-aligned).
    Outside: static hold = robust mean of T over the free segment. hybrid: keep T(t) when within the inlier gates of the rigid prediction."""
    n=len(T); who,_=grasp(T,hands,dmax=dmax); out=T.copy(); src=np.zeros(n,int)
    Hs={s:smooth_T(hands[s][0][:n],hand_half) for s in ('left','right')}
    for h,i,j in segments(who):
        if h<0:
            # free: robust static pose (median position, chordal-mean rotation with twins aligned to the segment's first frame)
            ref=T[i,:3,:3]; Rs=[]
            for k in range(i,j):
                Rs.append(min((T[k]@S for S in twins), key=lambda C: ang(C[:3,:3],ref))[:3,:3])
            Rm=mean_rot(np.array(Rs)); pm=np.median(T[i:j,:3,3],0)
            for k in range(i,j): out[k][:3,:3]=Rm; out[k][:3,3]=pm
            continue
        H=Hs[('left','right')[h]]; Xs=[np.linalg.inv(H[k])@T[k] for k in range(i,j)]
        # twin-align the relative transforms to the medoid
        Rrel=np.array([X[:3,:3] for X in Xs]); med=Rrel[int(np.argmin([sum(ang(a,b) for b in Rrel) for a in Rrel]))]
        al=[]
        for k,X in zip(range(i,j),Xs):
            C=min((X@S for S in twins), key=lambda C: ang(C[:3,:3],med)); al.append(C)
        al=np.array(al); Xr=np.eye(4); Xr[:3,:3]=mean_rot(al[:,:3,:3]); Xr[:3,3]=np.median(al[:,:3,3],0)
        for k in range(i,j):
            P=H[k]@Xr
            if hybrid:
                C=min((T[k]@S for S in twins), key=lambda C: ang(C[:3,:3],P[:3,:3]))
                if ang(C[:3,:3],P[:3,:3])<inlier_deg and np.linalg.norm(C[:3,3]-P[:3,3])*100<inlier_cm: out[k]=C; src[k]=2; continue
            out[k]=P; src[k]=1
    return out, who, src

def chain_hold(T, twins, hands, dmax=0.13, hand_half=3, static_frames=10, free_mode='hold', gap_merge=8):
    """Chained rigid hold. Initial pose = robust mean of T over the first static_frames (object at rest).
    Grasp segment: X = H(s)^-1 @ pose(s-1); pose(t) = H(t) @ X. Free segment: hold the previous pose (free_mode='hold')
    or re-acquire from T with twin chosen nearest the held pose (free_mode='reacquire'). Short free gaps between grasps by the
    same hand (< gap_merge frames) are merged into the grasp."""
    n=len(T); who,_=grasp(T,hands,dmax=dmax)
    segs=segments(who)
    for a in range(1,len(segs)-1):
        h,i,j=segs[a]
        if h<0 and j-i<gap_merge and segs[a-1][0]==segs[a+1][0]>=0: who[i:j]=segs[a-1][0]
    Hs={s:smooth_T(hands[s][0][:n],hand_half) for s in ('left','right')}
    out=np.empty_like(T); k0=max(1,min(static_frames,n))
    ref=T[0,:3,:3]; Rs=[min((T[k]@S for S in twins), key=lambda C: ang(C[:3,:3],ref))[:3,:3] for k in range(k0)]
    P0=np.eye(4); P0[:3,:3]=mean_rot(np.array(Rs)); P0[:3,3]=np.median(T[:k0,:3,3],0)
    prev=P0
    for h,i,j in segments(who):
        if h<0:
            for k in range(i,j):
                if free_mode=='reacquire':
                    C=min((T[k]@S for S in twins), key=lambda C: ang(C[:3,:3],prev[:3,:3]))
                    if ang(C[:3,:3],prev[:3,:3])<25 and np.linalg.norm(C[:3,3]-prev[:3,3])<0.03: prev=C
                out[k]=prev
            continue
        H=Hs[('left','right')[h]]; X=np.linalg.inv(H[i])@prev
        for k in range(i,j): out[k]=H[k]@X
        prev=out[j-1]
    return out, who

def combine(Trot, Tpos, c):
    """rotation from Trot, object-centre position (mesh centre c) from Tpos."""
    out=Trot.copy(); ctr=np.einsum('nij,j->ni',Tpos[:,:3,:3],c)+Tpos[:,:3,3]
    out[:,:3,3]=ctr-np.einsum('nij,j->ni',Trot[:,:3,:3],c); return out
