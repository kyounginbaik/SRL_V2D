import numpy as np,json,glob,os,trimesh,sys
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import median_filter
B='/z/'
def up(zp,h):
    W=zp['world_from_rgb'];C=h['camera_to_world_transform'];n=len(W);u=np.mean([W[i,:3,:3]@C[int(i*(len(C)-1)/max(1,n-1)),:3,:3].T@np.array([0,0,1.]) for i in range(n)],0);return u/np.linalg.norm(u)
def segs(lab):
    o=[];s=0
    for i in range(1,len(lab)+1):
        if i==len(lab) or lab[i]!=lab[s]:
            o.append((int(lab[s]),s,i-1));s=i
    return o
NM={-1:'none',0:'left',1:'right'}
for ep in sorted(os.listdir(B+'targets')):
    n=int(ep[-6:]);zp=np.load(B+'bundles/'+ep+'/object_poses.npz');names=[str(x) for x in zp['names']];T0=[np.load(B+f'targets/{ep}/{o}_trajectory.npz')['T_world_obj'] for o in names]
    print(f'ep{n} N={len(T0[0])}')
    for k,o in enumerate(names):
        T=T0[k];mv=100*np.linalg.norm(T[:,:3,3]-T[0,:3,3],axis=1);turn=np.degrees((R.from_matrix(T[:,:3,:3])*R.from_matrix(T[0,:3,:3]).inv()).magnitude())
        if mv.max()<2 and turn.max()<20:continue
        h=dict(np.load(B+f'hands/{ep}/hawor_{o}_result.npz'));sc=2/(float(h['hand_left_scale'].mean())+float(h['hand_right_scale'].mean()));u=up(zp,h);R1=R.align_vectors([[0,0,1.]],[u])[0];tp=T[:,:3,3];om=h['object_to_world_transform'][:,:3,3].astype(float)*sc;d1=R1.apply(tp-tp[0]);d2=om-om[0];yaw=np.arctan2(np.sum(d2[:,1]*d1[:,0]-d2[:,0]*d1[:,1]),np.sum(d2[:,0]*d1[:,0]+d2[:,1]*d1[:,1]));RA=(R.from_euler('z',yaw)*R1);Rr=RA*R.from_matrix(T[:,:3,:3]);P=RA.apply(tp)+om[0]-RA.apply(tp[0])
        vv=np.asarray(trimesh.load(B+f'targets/{ep}/{o}_mesh_metric.obj',force='mesh').vertices);vv=vv[::max(1,len(vv)//300)];D=[]
        for sd in('left','right'):
            W=h[f'hand_{sd}_wrist_to_world_transform'][:,:3,3].astype(float)*sc
            D.append(np.array([np.linalg.norm(Rr[t].apply(vv)+P[t]-W[t],axis=1).min() for t in range(len(T))])*100)
        D=np.array(D);lab=np.where(D.min(0)<12,D.argmin(0),-1);lab=median_filter(lab,size=9,mode='nearest');S=[(NM[a],b,c) for a,b,c in segs(lab) if c-b>=8]
        on=int(np.flatnonzero(mv>2)[0]) if (mv>2).any() else 0
        p=B+f'hands/{ep}/hand_motion_prior_v2.json';pr=''
        if os.path.exists(p):
            pj=json.load(open(p))['objects'].get(o);pr=[(s['hand'],s['start'],s['stop']) for s in pj['segments']] if pj else ''
        print(f'  {o} move {mv.max():.0f}cm turn {turn.max():.0f} onset {on} mine {S}'+(f'\n     prior {pr}' if pr else ''))
