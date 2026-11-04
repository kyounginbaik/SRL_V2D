import sys,numpy as np,cv2
from scipy.spatial.transform import Rotation as R
T_='cw_target/'
sys.argv=['r','x','balls=1'];src=open('refine.py').read().split("if __name__")[0];exec(src)
Z2=np.load(f'{KD}/object_poses.npz');WR=Z2['world_from_rgb'];iW=np.linalg.inv(WR)
Ths=np.einsum('nij,njk->nik',iW,np.load(T_+'wooden_spoon_trajectory.npz')['T_world_obj']);Thc=np.einsum('nij,njk->nik',iW,np.load(T_+'blue_cup_trajectory.npz')['T_world_obj'])
Ns=np.load('full_0.1.npz')['T'];Nh=np.load('hold_0.1.npz')['T'];Nc=np.load('cupfull_0.1.npz')['T'];Oc=Z2['camera_from_object'][:,list(Z2['names']).index('blue_cup')]
SC=0.5
def score(Ts,dat_fn):return np.array([parts(Ts[p],*dat_fn(p)) for p in range(len(Ts)-1)])
def ds(p):
    Mk,h,D,det=load(p+1);c=T0[p][:3,:3]@com+T0[p][:3,3];return Mk,h,D,[d for d in det if np.linalg.norm(np.asarray(d['camera_point'])-c)<.15]
Pa=np.array([[parts(X[p],*ds(p)) for X in (T0,Ths,Ns,Nh)] for p in range(258)]);np.save('cmp_spoon.npy',Pa)
# cup scored with the cup mask and hand+spoon occluders
g={};exec(open('refine_cup.py').read().split("if __name__")[0].replace("sys.argv[2]","'balls=1'"),g);g['SC']=.5
def dc(p):
    Mk,h,D,det=g['load'](p+1);c=Oc[p][:3,:3]@g['com']+Oc[p][:3,3];return Mk,h,D,[d for d in det if np.linalg.norm(np.asarray(d['camera_point'])-c)<.15]
Pc=np.array([[g['parts'](X[p],*dc(p)) for X in (Oc,Thc,Nc)] for p in range(258)]);np.save('cmp_cup.npy',Pc)
def rate(T):
    r=R.from_matrix(T[:,:3,:3]);return np.degrees((r[:-1].inv()*r[1:]).magnitude())
def wr(C_):return np.einsum('nij,njk->nik',WR,C_)
def cm(T,c):
    W=wr(T);x=W[:,:3,:3]@c+W[:,:3,3];return 100*np.linalg.norm(np.diff(x,axis=0),axis=1)
print('SPOON  old | hansol | ours')
for nm,R0 in (('all p>=35',range(35,258)),('f85-95',range(85,96)),('f148-157',range(148,158)),('f170-185',range(170,186))):
    r=list(R0);print(nm,'iou',np.round(Pa[r,:,1].mean(0)[:3],3),'min',np.round(Pa[r,:,1].min(0)[:3],2),'ball px',np.round(np.nanmean(Pa[r,:,3],axis=0)[:3],1),'depth mm',np.round(1e3*Pa[r,:,2].mean(0)[:3],1))
h=[p for p in range(35,258) if p%3==1 and np.isfinite(Pa[p,0,3])];print('held-out ball frames',len(h),'old %.1f hansol %.1f ours(held-out fit) %.1f'%(np.nanmean(Pa[h,0,3]),np.nanmean(Pa[h,1,3]),np.nanmean(Pa[h,3,3])),'med',np.round(np.nanmedian(Pa[h][:,[0,1,3],3],axis=0),1))
for n,X in (('old',T0),('hansol',Ths),('ours',Ns)):
    q=rate(wr(X));print(n,'step deg mean %.1f max %.0f >30:%d >15:%d'%(q.mean(),q.max(),(q>30).sum(),(q>15).sum()),' worst',[(int(i),round(float(q[i]))) for i in np.argsort(-q)[:6]])
print('CUP old | hansol | ours (cup mask)')
for nm,R0 in (('all p>=6',range(6,258)),('f125-165',range(125,165))):
    r=list(R0);print(nm,'iou',np.round(Pc[r,:,1].mean(0),3),'min',np.round(Pc[r,:,1].min(0),2),'ball px',np.round(np.nanmean(Pc[r,:,3],axis=0),1),'depth mm',np.round(1e3*Pc[r,:,2].mean(0),1))
for n,X in (('old',Oc),('hansol',Thc),('ours',Nc)):
    q=rate(wr(X));s=cm(X,g['com']);print(n,'step deg mean %.1f max %.0f >15:%d | step cm mean %.2f max %.2f >2cm:%d'%(q.mean(),q.max(),(q>15).sum(),s.mean(),s.max(),(s>2).sum()))
# spoon-cup penetration
import trimesh
mc=g['m'];sp=fc
def pen(TS,TC):
    from trimesh.proximity import signed_distance
    out=np.zeros(258);pts=v[np.random.default_rng(0).choice(len(v),700,replace=False)]
    for p in range(0,258,2):
        Rel=np.linalg.inv(TC[p])@TS[p];q=pts@Rel[:3,:3].T+Rel[:3,3];d=mc.nearest.signed_distance(q);out[p]=max(d.max(),0)*100
    return out
print('cup watertight',mc.is_watertight)
for n,a,b in (('old',T0,Oc),('hansol',Ths,Thc),('ours',Ns,Nc)):
    q=pen(a,b);print(n,'spoon into cup wall cm: mean %.2f max %.2f frames>0.5cm %d; 4-5s max %.2f 7-8s max %.2f'%(q[::2].mean(),q.max(),(q[::2]>.5).sum()*2,q[80:100].max(),q[140:160].max()))
np.savez('cmp_tracks.npz',Ths=Ths,Thc=Thc,Ns=Ns,Nc=Nc,T0=T0,Oc=Oc)
