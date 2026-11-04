import json,time,glob,sys,numpy as np,trimesh,coacd,os
from scipy.spatial.transform import Rotation as RR
from scipy.spatial import ConvexHull
n=16;e=f'/uniretarget_yam/episodes/recon_{n:06d}';o=f'/uniretarget_yam/objects/recon_{n:06d}';REG=f'/scratch2/exaFLOPs/srl_yam_run/source_meshes/registered/episode_{n:06d}/'
z=np.load(e+'/baseline/physics.npz');P=z['object_pose'][:,0]
def frame(i,f):
    T=np.eye(4);T[:3,3]=P[f,i,7:10];T[:3,:3]=RR.from_quat(P[f,i,10:14]).as_matrix();return T
spoon=[trimesh.load(p,force='mesh') for p in sorted(glob.glob(o+'/wooden_spoon_collision/part_*.obj'))];sv=np.vstack([s.vertices for s in spoon]);sv=sv[::max(1,len(sv)//400)]
def depth(hulls,fs=(110,130,150,170)):
    eq=[ConvexHull(h.vertices).equations for h in hulls];out=[]
    for f in fs:
        T=np.linalg.inv(frame(0,f))@frame(1,f);v=(T[:3,:3]@sv.T).T+T[:3,3];d=0.
        for E in eq:
            s=(E[:,:3]@v.T).T+E[:,3];inside=s.max(1);d=max(d,float(-inside.min()))
        out.append(round(100*d,2))
    return out
cur=[trimesh.load(p,force='mesh') for p in sorted(glob.glob(o+'/plastic_dish_rack_collision/part_*.obj'))];print('current hulls',len(cur),'spoon depth cm at f110,130,150,170',depth(cur),flush=True)
src=REG+'plastic_dish_rack_decomp.obj';m=trimesh.load(src,force='mesh');full=trimesh.load(REG+'plastic_dish_rack.obj',force='mesh',process=False);print('public mesh faces',len(full.faces),'watertight',full.is_watertight,flush=True)
# Is the holder hollow in the public mesh? Slice through the spoon handle's location at frame 150.
T=np.linalg.inv(frame(0,150))@frame(1,150);v=(T[:3,:3]@sv.T).T+T[:3,3];z0=np.percentile(v[:,2],25);c=np.median(v[v[:,2]<np.percentile(v[:,2],30)],axis=0);print('spoon low part in rack frame',c.round(3),flush=True)
np.savez('/scratch2/exaFLOPs/srl_yam_run/videos/diagnosis/.rack_slice.npz',v=v,c=c)
for th,mx in ((.03,32),(.02,64),(.01,128)):
    t=time.time();parts=coacd.run_coacd(coacd.Mesh(m.vertices,m.faces),threshold=th,max_convex_hull=mx,preprocess_mode='auto');H=[trimesh.Trimesh(a,b).convex_hull for a,b in parts];print('threshold',th,'max',mx,'hulls',len(H),'sec',round(time.time()-t),'spoon depth cm',depth(H),flush=True)
    os.makedirs(f'/scratch2/exaFLOPs/srl_yam_run/videos/diagnosis/.rack_{th}',exist_ok=True)
    for i,h in enumerate(H):h.export(f'/scratch2/exaFLOPs/srl_yam_run/videos/diagnosis/.rack_{th}/part_{i:03d}.obj')
