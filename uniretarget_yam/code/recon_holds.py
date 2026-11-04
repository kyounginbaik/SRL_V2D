"""Which hand holds the moving object, until when, and how much the goal must be conditioned for the arm.
Everything comes from the reconstruction: no per-episode choices."""
from pathlib import Path
import numpy as np,trimesh
from scipy.ndimage import gaussian_filter1d,median_filter
from scipy.spatial.transform import Rotation as R
SWITCH_CM=5.;NEAR_CM,STILL_CM,STILL_DEG,MIN_FRAMES,AFTER=12.,3.,25.,8,5;SPEED=172.;SIGMAS=(1.5,3.,4.,6.,8.,10.)
def segments(lab):
    o=[];s=0
    for i in range(1,len(lab)+1):
        if i==len(lab) or lab[i]!=lab[s]:
            if i-s>=MIN_FRAMES:o.append((int(lab[s]),s,i-1))
            s=i
    return o
def wrist_surface_cm(T,mesh_path,H,up):
    """Wrist to object surface per frame, in the target world. The hand run's own object track can be metres off, so its world is tied to the target world through the camera both share."""
    B=Path(mesh_path).parents[2]/'bundles'/Path(mesh_path).parent.name/'object_poses.npz';W=np.load(B)['world_from_rgb'];C=H['camera_to_world_transform'].astype(float);n=min(len(T),len(C),len(W));sc=2/(float(H['hand_left_scale'].mean())+float(H['hand_right_scale'].mean()))
    Rm=R.from_matrix(np.array([W[i,:3,:3]@C[i,:3,:3].T for i in range(n)])).mean();t=np.median(W[:n,:3,3]-Rm.apply(C[:n,:3,3]*sc),axis=0)
    Rr=R.from_matrix(T[:n,:3,:3]);P=T[:n,:3,3];v=np.asarray(trimesh.load(mesh_path,force='mesh').vertices);v=v[::max(1,len(v)//300)];D=[]
    for sd in('left','right'):
        w=Rm.apply(H[f'hand_{sd}_wrist_to_world_transform'][:n,:3,3].astype(float)*sc)+t;d=np.array([np.linalg.norm(Rr[f].apply(v)+P[f]-w[f],axis=1).min() for f in range(n)])*100
        # Before the first and after the last detection the hand file only holds a pose: that is not the hand being near anything.
        ok=np.flatnonzero(H[f'hand_{sd}_is_valid'][:n].astype(bool))
        if len(ok):d[:ok[0]]=999.;d[ok[-1]+1:]=999.
        D.append(np.r_[d,np.full(len(T)-n,d[-1])])
    return np.array(D)
def plan(tg,hd,names,T,up,mk=None):
    """T: (F,K,4,4). The mover, the hand holding it at lift-off, and the frame to stop at if another hand takes over while it still moves."""
    F=len(T);mv=100*np.linalg.norm(T[:,:,:3,3]-T[0,:,:3,3],axis=-1);mk=int(mv.max(0).argmax()) if mk is None else mk;n=names[mk];X=T[:,mk];moved=mv[:,mk]>2;turn=np.degrees((R.from_matrix(X[:,:3,:3])*R.from_matrix(X[0,:3,:3]).inv()).magnitude())
    onset=int(np.flatnonzero(moved)[0]) if moved.any() else int(np.flatnonzero(turn>20)[0])
    H=dict(np.load(hd/f'hawor_{n}_result.npz'));D=wrist_surface_cm(X,tg/f'{n}_mesh_metric.obj',H,up);lab=np.where(D.min(0)<NEAR_CM,D.argmin(0),-1);lab=median_filter(lab,size=9,mode='nearest');segs=segments(lab)
    near=[s for s in segs if s[0]>=0 and s[1]<=onset+10 and s[2]>=onset-10] or [s for s in segs if s[0]>=0 and s[2]>=onset] or [(int(D[:,onset].argmin()),onset,onset)]
    hand=max(near,key=lambda s:min(s[2],onset+30)-max(s[1],onset-10))[0];start=max(near,key=lambda s:min(s[2],onset+30)-max(s[1],onset-10))[2];stop=start;handoff=None
    for s in segs:
        if s[2]<=start or s[0]<0:continue
        if s[0]==hand:stop=s[2]
        else:handoff=s[1];break
    Ro=R.from_matrix(X[:,:3,:3]);after=np.arange(stop,F);still=bool((100*np.linalg.norm(X[after,:3,3]-X[stop,:3,3],axis=1)).max()<STILL_CM and np.degrees((Ro[after]*Ro[stop].inv()).magnitude()).max()<STILL_DEG)
    end=F-1 if still else min(F-1,stop+AFTER)
    return dict(mover=mk,object=n,side=('left','right')[hand],onset=onset,hold_end=int(stop),handoff=handoff,end=int(end),full_frames=F,coverage=round((end+1)/F,3),still_after_hold=still,segments=[(('none','left','right')[a+1],b,c) for a,b,c in segs])
SYM_MAX=.06
def symmetry(path):
    """The mesh axis it can be spun about and still look the same, or None. Worst surface mismatch over spins, relative to the radius."""
    import trimesh
    from scipy.spatial import cKDTree
    m=trimesh.load(path,force='mesh');P,_=trimesh.sample.sample_surface(m,6000,seed=0);c=P.mean(0);X=P-c;_,V=np.linalg.eigh(X.T@X/len(X));best=None;t=cKDTree(X)
    for i in range(3):
        a=V[:,i];rad=np.percentile(np.linalg.norm(X-np.outer(X@a,a),axis=1),95)
        e=max(t.query(R.from_rotvec(a*np.radians(d)).apply(X))[0].mean() for d in (30,60,90,120,150,180))/rad
        if best is None or e<best[0]:best=(float(e),a)
    return dict(axis=best[1].tolist(),center=c.tolist(),mismatch=round(best[0],3)) if best[0]<=SYM_MAX else None
def despin(p,r,axis,center):
    """Spin about a symmetry axis cannot be seen, so tracked spin is noise no hand should follow. Keep where the axis points and where the object is; drop the spin."""
    a,c=np.asarray(axis),np.asarray(center);d=r.apply(a);out=[R.identity()]*len(d);out[0]=r[0]
    for t in range(1,len(d)):
        x=d[t-1];y=d[t];v=np.cross(x,y);ang=np.arctan2(np.linalg.norm(v),x@y);out[t]=(R.from_rotvec(v/max(np.linalg.norm(v),1e-9)*ang) if np.linalg.norm(v)>1e-9 else R.identity())*out[t-1]
    Q=R.concatenate(out);dR=(r[:-1].inv()*r[1:]).as_rotvec()@a
    return p+r.apply(c)-Q.apply(c),Q,dict(spin_path_deg=round(float(np.degrees(np.abs(dR).sum())),1),spin_range_deg=round(float(np.ptp(np.degrees(np.cumsum(dR)))) if len(dR) else 0.,1))
def condition(p,r,speed=SPEED):
    """Remove single-frame position glitches, then the least smoothing that brings rotation under the arm's speed."""
    q=r.as_quat()
    for t in range(1,len(q)):
        if q[t]@q[t-1]<0:q[t]=-q[t]
    def rot(s):
        y=gaussian_filter1d(q,s,axis=0,mode='nearest');return R.from_quat(y/np.linalg.norm(y,axis=1,keepdims=True))
    sigma=SIGMAS[-1]
    for s in SIGMAS:
        x=rot(s)
        if len(x)<3 or np.percentile(np.degrees((x[1:]*x[:-1].inv()).magnitude())*20,99)<=speed:sigma=s;break
    if sigma==SIGMAS[0]:P,Q=p.copy(),r
    else:P=gaussian_filter1d(median_filter(p,size=(7,1),mode='nearest'),sigma,axis=0,mode='nearest');Q=rot(sigma)
    glitch=np.linalg.norm(p-median_filter(p,size=(7,1),mode='nearest'),axis=1)>.04
    if sigma==SIGMAS[0] and glitch.any():P=median_filter(p,size=(7,1),mode='nearest')
    dev=np.degrees((Q*r.inv()).magnitude());dp=100*np.linalg.norm(P-p,axis=1)
    return P,Q,dict(sigma=None if sigma==SIGMAS[0] else sigma,glitch_frames=int(glitch.sum()),dev_deg_mean=round(float(dev.mean()),1),dev_deg_p95=round(float(np.percentile(dev,95)),1),dev_deg_max=round(float(dev.max()),1),dev_cm_mean=round(float(dp.mean()),2),dev_cm_max=round(float(dp.max()),2))

TASK_CM,TASK_DEG,MIN_HOLD=3.,45.,20
def plan_all(tg,hd,names,T,up):
    """The primary hold, plus another object carried by the other hand. Objects no free hand can carry stay where they start."""
    first=plan(tg,hd,names,T,up);F=first['end']+1;holds=[dict(index=first['mover'],object=first['object'],side=first['side'],onset=first['onset'],end=first['end'])];rest=[]
    for k in range(T.shape[1]):
        if k==first['mover']:continue
        X=T[:F,k];mv=100*np.linalg.norm(X[:,:3,3]-X[0,:3,3],axis=1).max();tr=np.degrees((R.from_matrix(X[:,:3,:3])*R.from_matrix(X[0,:3,:3]).inv()).magnitude()).max()
        if mv<TASK_CM and tr<TASK_DEG:rest.append(k);continue
        h=plan(tg,hd,names,T,up,mk=k);end=min(h['end'],F-1)
        if h['side']==first['side'] or len(holds)>1 or end<h['onset']+MIN_HOLD:rest.append(k);continue
        holds.append(dict(index=k,object=h['object'],side=h['side'],onset=h['onset'],end=int(end)))
    return first,holds,rest

GAP,MIN_LEN,COHOLD_MIN,MAX_STAGES=20,20,40,6
def _intervals(mask):
    o=[];i=0;n=len(mask)
    while i<n:
        if mask[i]:
            j=i
            while j+1<n and mask[j+1]:j+=1
            o.append([i,j]);i=j+1
        else:i+=1
    m=[]
    for a in o:
        if m and a[0]-m[-1][1]<=GAP:m[-1][1]=a[1]
        else:m.append(a)
    return [x for x in m if x[1]-x[0]+1>=MIN_LEN]
def schedule(tg,hd,names,T,up):
    """Contact schedule: which hand holds which object from when to when. One object in one hand, two objects, handover, regrasp and a both-arm carry are all just overlaps of these intervals."""
    F,K=T.shape[:2];Ro=[R.from_matrix(T[:,k,:3,:3]) for k in range(K)];mv=100*np.linalg.norm(T[:,:,:3,3]-T[0,:,:3,3],axis=-1);tr=np.array([np.degrees((Ro[k]*Ro[k][0].inv()).magnitude()) for k in range(K)]).T;task=[k for k in range(K) if mv[:,k].max()>=TASK_CM or tr[:,k].max()>=TASK_DEG]
    if not task:task=[int(mv.max(0).argmax())]
    D={k:wrist_surface_cm(T[:,k],tg/f'{names[k]}_mesh_metric.obj',dict(np.load(hd/f'hawor_{names[k]}_result.npz')),up) for k in task};holds=[]
    # A hand beside a still object while another object moves is holding the one that moves.
    w=15;active={}
    for k in task:
        sp=np.r_[0.,100*np.linalg.norm(np.diff(T[:,k,:3,3],axis=0),axis=1)*20];av=np.r_[0.,np.degrees((Ro[k][1:]*Ro[k][:-1].inv()).magnitude())*20];active[k]=np.convolve((sp>3)|(av>15),np.ones(2*w+1),'same')>0
    # A grip persists until the hand has clearly left the object, or the other hand has it and this one has moved off. A free hand takes the moving object first.
    labs=np.full((2,F),-1);cur=[-1,-1]
    for f in range(F):
        for h in (0,1):
            A=np.array([D[k][h,f] for k in task]);O=np.array([D[k][1-h,f] for k in task])
            if cur[h]>=0 and (A[cur[h]]>NEAR_CM*1.5 or (A[cur[h]]>NEAR_CM and O[cur[h]]<NEAR_CM) or A[cur[h]]-A.min()>SWITCH_CM):cur[h]=-1  # Clearly closer to another object: it has moved on.
            if cur[h]<0:
                c=A+np.array([100.*~active[k][f] for k in task]);i=int(c.argmin());j=int(A.argmin());cur[h]=i if A[i]<NEAR_CM else (j if A[j]<NEAR_CM else -1)
            labs[h,f]=cur[h]
    for h,side in enumerate(('left','right')):
        lab=median_filter(labs[h],size=9,mode='nearest')
        for i,k in enumerate(task):
            for s,e in _intervals(lab==i):holds.append(dict(index=k,object=names[k],side=side,start=int(s),end=int(e)))
    # The same hand keeps its grip across a gap unless the other hand held the object meanwhile.
    holds.sort(key=lambda x:(x['side'],x['index'],x['start']));merged=[]
    for b in holds:
        a=merged[-1] if merged else None
        if a and a['side']==b['side'] and a['index']==b['index'] and not any(c['index']==b['index'] and c['side']!=b['side'] and c['end']>a['end']+MIN_LEN//2 and c['start']<b['start']-MIN_LEN//2 and c['start']>a['start'] for c in holds):a['end']=b['end']
        else:merged.append(dict(b))
    # A hand holds one object at a time, and only an object that moves while it does.
    holds=[]
    for b in merged:
        dp=100*np.linalg.norm(T[b['start']:b['end']+1,b['index'],:3,3]-T[b['start'],b['index'],:3,3],axis=1).max();dr=np.degrees((Ro[b['index']][b['start']:b['end']+1]*Ro[b['index']][b['start']].inv()).magnitude()).max()
        if dp>=STILL_CM or dr>=STILL_DEG:holds.append(b)
    holds.sort(key=lambda x:x['start']);last={}
    for b in holds:
        p=last.get(b['side'])
        if p is not None and p['end']>=b['start']:p['end']=b['start']-1
        last[b['side']]=b
    holds=[b for b in holds if b['end']-b['start']+1>=MIN_LEN]
    holds.sort(key=lambda x:x['start'])
    # A brief overlap with the object's current holder is a hand passing nearby, not a grip; a long one is a both-arm carry.
    keep=[]
    for b in holds:
        over=[min(a['end'],b['end'])-max(a['start'],b['start'])+1 for a in keep if a['index']==b['index'] and a['side']!=b['side']];ov=max(over) if over else 0
        if ov>=.8*(b['end']-b['start']+1) and ov<COHOLD_MIN:continue
        keep.append(b)
    keep=keep[:MAX_STAGES];out=[];ends={}
    for k in task:
        hk=[x for x in keep if x['index']==k]
        if not hk:continue
        moves=np.flatnonzero((mv[:,k]>2)|(tr[:,k]>STILL_DEG));first=int(moves[0]) if len(moves) else hk[0]['start'];last=max(x['end'] for x in hk)
        after=np.arange(last,F);still=bool((100*np.linalg.norm(T[after,k,:3,3]-T[last,k,:3,3],axis=1)).max()<STILL_CM and np.degrees((Ro[k][after]*Ro[k][last].inv()).magnitude()).max()<STILL_DEG)
        ends[k]=F-1 if still else min(F-1,last+AFTER)
        for i,x in enumerate(hk):
            x['onset']=first if i==0 else x['start'];x['join']=i>0 or x['start']>first+10
    primary=int(max(ends,key=lambda k:mv[:ends[k]+1,k].max()));end=ends[primary]
    for x in sorted(keep,key=lambda x:(x['onset'],x['start'])):
        if x['start']>end-MIN_LEN//2:continue
        x['object_end']=min(ends[x['index']],end);x['end']=min(x['end'],end);out.append(x)
    for i,x in enumerate(out):
        nxt=[y for y in out[i+1:] if y['index']==x['index'] and y['side']!=x['side'] and y['start']>x['start']]
        own=[y for y in out[i+1:] if y['side']==x['side']]
        x['released_by_next']=bool(nxt and x['end']<x['object_end']) ;x['release']=int(max(x['end'],nxt[0]['start']+4)) if (nxt and x['end']<x['object_end']) else (int(x['end']) if own else None)
    frozen=[k for k in range(K) if k not in {x['index'] for x in out}]
    kind=[];sides={}
    for x in out:sides.setdefault(x['index'],[]).append(x)
    if len(sides)>1:kind.append('two objects')
    for k,v in sides.items():
        for a,b in zip(v,v[1:]):
            ov=min(a['end'],b['end'])-max(a['start'],b['start'])+1
            kind.append('both-arm carry' if ov>=COHOLD_MIN and a['side']!=b['side'] else 'handover' if a['side']!=b['side'] else 'regrasp')
    # An object nobody holds keeps its recorded path and still counts; say so whenever it really moved, since no hand is there to move it.
    untracked=[dict(object=names[k],move_cm=round(float(mv[:,k].max()),1),turn_deg=round(float(tr[:,k].max()),1)) for k in frozen if mv[:,k].max()>=STILL_CM or tr[:,k].max()>=STILL_DEG]
    return dict(primary=primary,end=int(end),full_frames=F,coverage=round((end+1)/F,3),holds=out,frozen=frozen,untracked=untracked,kinds=sorted(set(kind)) or ['one hand'],stages=len(out))

LEAD_IN=40
def pad_schedule(sc):
    """A robot starting from its ready pose needs time before the first object moves: the episode starts with everything held still."""
    first=min(x['onset'] for x in sc['holds']);pad=max(0,LEAD_IN-first)
    for x in sc['holds']:
        for k in('start','end','onset','object_end'):x[k]=int(x[k])+pad
        if x.get('release') is not None:x['release']=int(x['release'])+pad
    sc['end']+=pad;sc['pad']=pad;return pad
